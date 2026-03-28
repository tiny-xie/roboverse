from __future__ import annotations

import logging
import os
import re
import sys
import time
import json
from pathlib import Path
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio as iio
import numpy as np
import torch
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid, save_image
def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
from metasim.scenario.cameras import PinholeCameraCfg

# from metasim.scenario.randomization import RandomizationCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.robot import RobotCfg
from metasim.task.registry import get_task_class
from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "libero_90.kitchen_scene1_open_drawer_put_bowl"
    robot: str = "franka"
    scene: str | None = None
    # Native LIBERO demos are rendered without raytracing; this avoids overly bright scenes.
    render: RenderCfg = RenderCfg(mode="rasterization")
    # random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "test_output/tmp"
    save_video_path: str | None = "test_output/test_replay.mp4"
    stop_on_runout: bool = False

    # Action source:
    # - traj: use native demo joint actions from trajectory file (default behavior)
    # - action_md: parse logged EE delta actions from markdown and convert via IK
    # - action_json: load EE delta actions from JSON and convert via IK
    action_source: Literal["traj", "action_md", "action_json"] = "action_json"
    action_md_path: str = "action.md"
    action_json_path: str = (
        "/home/x/robot_manipulation/opensource/RoboVerse/scripts/roboverse_data/trajs/libero10/"
        "actions7d_dataset_20260328_164711_acf73fdd.json"
    )
    action_json_episode_index: int = 0
    ee_action_coordinate_frame: Literal["local", "world"] = "world"
    ik_solver: Literal["pyroki", "curobo"] = "pyroki"

    # Camera controls for replay visualization.
    camera_preset: Literal["default", "libero_front"] = "default"
    camera_pos: tuple[float, float, float] | None = None
    camera_look_at: tuple[float, float, float] | None = None

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


###########################################################
## Utils
###########################################################
def get_actions(all_actions, action_idx: int, num_envs: int, robot: RobotCfg):
    envs_actions = all_actions[:num_envs]
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


def get_states(all_states, action_idx: int, num_envs: int):
    envs_states = all_states[:num_envs]
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def get_runout(all_actions, action_idx: int):
    runout = all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])
    return runout


def _resolve_data_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    script_root = Path(__file__).resolve().parents[2]  # <repo>/RoboVerse
    candidate = script_root / path_str
    if candidate.exists():
        return candidate

    return path


def load_actions_from_markdown(path: str) -> list[np.ndarray]:
    """Parse lines like: actions=[[x y z rx ry rz g]] from markdown/log text."""
    action_path = _resolve_data_path(path)
    if not action_path.exists():
        raise FileNotFoundError(f"Action markdown file not found: {action_path}")

    text = action_path.read_text(encoding="utf-8")
    matches = re.findall(r"actions=\[\[([^\]]+)\]\]", text)
    parsed: list[np.ndarray] = []
    for raw in matches:
        values = np.fromstring(raw.replace(",", " "), sep=" ", dtype=np.float32)
        if values.size < 7:
            continue
        parsed.append(values[:7])

    if not parsed:
        raise ValueError(f"No valid 'actions=[[...]]' entries found in {action_path}")
    return parsed


def _parse_actions7d(raw_actions: list) -> list[np.ndarray]:
    parsed: list[np.ndarray] = []
    for row in raw_actions:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size < 7:
            continue
        parsed.append(values[:7])
    return parsed


def load_actions_from_json(path: str, episode_index: int = 0) -> tuple[list[np.ndarray], str]:
    """Load 7D EE delta actions from a JSON file.

    Supported formats:
    - {"episodes": [{"action_7d": [[...], ...]}, ...]}
    - {"action_7d": [[...], ...]}
    - [[...], ...]
    """
    action_path = _resolve_data_path(path)
    if not action_path.exists():
        raise FileNotFoundError(f"Action JSON file not found: {action_path}")

    payload = json.loads(action_path.read_text(encoding="utf-8"))
    source_desc = ""

    if isinstance(payload, dict) and "episodes" in payload:
        episodes = payload["episodes"]
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"JSON field 'episodes' is empty or invalid in {action_path}")

        if episode_index < 0 or episode_index >= len(episodes):
            raise IndexError(
                f"episode_index={episode_index} is out of range [0, {len(episodes) - 1}] for {action_path}"
            )

        episode = episodes[episode_index]
        raw_actions = episode.get("action_7d") if isinstance(episode, dict) else None
        if raw_actions is None:
            raise ValueError(f"Episode {episode_index} has no 'action_7d' field in {action_path}")
        source_desc = f"episode {episode_index}"
    elif isinstance(payload, dict) and "action_7d" in payload:
        raw_actions = payload["action_7d"]
        source_desc = "top-level action_7d"
    elif isinstance(payload, list):
        raw_actions = payload
        source_desc = "top-level list"
    else:
        raise ValueError(
            f"Unsupported JSON format in {action_path}. Expected episodes/action_7d/list style action payload."
        )

    parsed = _parse_actions7d(raw_actions)
    if not parsed:
        raise ValueError(f"No valid 7D actions found in {action_path} ({source_desc})")

    return parsed, source_desc


class EEActionIKConverter:
    """Convert EE delta actions (dx, dy, dz, dRx, dRy, dRz, gripper) to joint commands via IK."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        handler,
        device: torch.device,
        solver: Literal["pyroki", "curobo"],
        action_coordinate_frame: Literal["local", "world"],
    ):
        self.robot_cfg = robot_cfg
        self.robot_name = robot_cfg.name
        self.device = device
        self.handler = handler
        self.action_coordinate_frame = action_coordinate_frame

        self.ik_solver = setup_ik_solver(robot_cfg, solver)
        self.reorder_idx = self.handler.get_joint_reindex(self.robot_name)
        self.inverse_reorder_idx = [self.reorder_idx.index(i) for i in range(len(self.reorder_idx))]
        self.ee_body_name = robot_cfg.ee_body_name
        self.ee_body_idx: int | None = None

    def convert(self, action: torch.Tensor, obs: TensorState) -> list[dict]:
        from pytorch3d import transforms

        num_envs = action.shape[0]
        rs = obs.robots[self.robot_name]

        joint_pos_raw = rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)
        curr_robot_q = joint_pos_raw[:, self.inverse_reorder_idx].to(self.device).float()
        robot_ee_state = (
            rs.body_state if isinstance(rs.body_state, torch.Tensor) else torch.tensor(rs.body_state)
        ).to(self.device).float()
        robot_root_state = (
            rs.root_state if isinstance(rs.root_state, torch.Tensor) else torch.tensor(rs.root_state)
        ).to(self.device).float()

        if self.ee_body_idx is None:
            self.ee_body_idx = rs.body_names.index(self.ee_body_name)

        ee_p_world = robot_ee_state[:, self.ee_body_idx, 0:3]
        ee_q_world = robot_ee_state[:, self.ee_body_idx, 3:7]
        robot_pos = robot_root_state[:, 0:3]
        robot_quat = robot_root_state[:, 3:7]

        inv_base_q = transforms.quaternion_invert(robot_quat).to(dtype=torch.float32)
        curr_ee_pos_local = transforms.quaternion_apply(inv_base_q, ee_p_world - robot_pos).to(dtype=torch.float32)
        curr_ee_quat_local = transforms.quaternion_multiply(inv_base_q, ee_q_world).to(dtype=torch.float32)

        action = action.to(self.device).float()
        ee_pos_delta_raw = action[:num_envs, :3]
        ee_rot_delta_raw = action[:num_envs, 3:6]
        gripper_raw = action[:num_envs, 6]

        # Action limiting:
        # - translation delta in [-0.05, 0.05]
        # - rotation delta (Euler XYZ) in [-0.5, 0.5]
        # - gripper: value < 0 -> 1, else -> 0
        # ee_pos_delta_raw = torch.clamp(ee_pos_delta_raw, min=-0.02, max=0.02)
        # ee_rot_delta_raw = torch.clamp(ee_rot_delta_raw, min=-0.05, max=0.05)
        gripper_open = torch.where(
            gripper_raw < 0.0,
            torch.ones_like(gripper_raw, dtype=torch.float32),
            torch.zeros_like(gripper_raw, dtype=torch.float32),
        )

        if self.action_coordinate_frame == "world":
            ee_pos_delta = transforms.quaternion_apply(inv_base_q, ee_pos_delta_raw).to(dtype=torch.float32)
            ee_rot_matrix_world = transforms.euler_angles_to_matrix(ee_rot_delta_raw, "XYZ")
            ee_rot_quat_world = transforms.matrix_to_quaternion(ee_rot_matrix_world).to(dtype=torch.float32)
            ee_quat_delta = transforms.quaternion_multiply(inv_base_q, ee_rot_quat_world).to(dtype=torch.float32)
        else:
            ee_pos_delta = ee_pos_delta_raw
            ee_quat_delta = transforms.matrix_to_quaternion(
                transforms.euler_angles_to_matrix(ee_rot_delta_raw, "XYZ")
            ).to(dtype=torch.float32)

        ee_pos_target = (curr_ee_pos_local + ee_pos_delta).to(device=self.device, dtype=torch.float32)
        ee_quat_target = transforms.quaternion_multiply(curr_ee_quat_local, ee_quat_delta).to(
            device=self.device, dtype=torch.float32
        )

        q_solution, ik_succ = self.ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)
        if not ik_succ.all():
            log.warning(f"IK failed for {(~ik_succ).sum().item()}/{num_envs} envs; falling back to best-effort actions.")

        gripper_widths = process_gripper_command(gripper_open, self.robot_cfg, self.device)
        return self.ik_solver.compose_joint_action(
            q_solution,
            gripper_widths,
            current_q=curr_robot_q,
            return_dict=True,
        )


class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        """Add the observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


###########################################################
## Main
###########################################################
def main():
    no_display = not _has_display()
    if no_display and not args.headless:
        log.warning("No DISPLAY/WAYLAND detected. Forcing --headless=TRUE for replay.")
        args.headless = True
    task_cls = get_task_class(args.task)
    use_task_cameras = (
        bool(task_cls.scenario.cameras)
        and args.camera_pos is None
        and args.camera_look_at is None
        and args.camera_preset == "default"
    )

    cam_focal_length = 24.0

    if args.camera_pos is not None and args.camera_look_at is not None:
        cam_pos = args.camera_pos
        cam_look_at = args.camera_look_at
    elif args.camera_preset == "libero_front" or args.task.startswith("libero_"):
        # Use a slightly tighter framing on the tabletop workspace.
        cam_pos = (1.0, 0.0, 1.22)
        cam_look_at = (-0.02, 0.0, 0.64)
        cam_focal_length = 40.0
    else:
        cam_pos = (1.5, -1.5, 1.5)
        cam_look_at = (0.0, 0.0, 0.0)

    camera = PinholeCameraCfg(pos=cam_pos, look_at=cam_look_at, focal_length=cam_focal_length)
    cameras = task_cls.scenario.cameras if use_task_cameras else [camera]

    scene_cfg = task_cls.scenario.scene if task_cls.scenario.scene is not None else args.scene
    if scene_cfg is None:
        log.warning("Scene is not specified by task or args; proceeding with None.")

    if args.robot == "None":
        scenario = task_cls.scenario.update(
            # robots=[args.robot],
            scene=scene_cfg,
            cameras=cameras,
            # random=args.random,
            render=args.render,
            simulator=args.sim,
            renderer=args.renderer,
            num_envs=args.num_envs,
            headless=args.headless,
        )

    else:
        scenario = task_cls.scenario.update(
            robots=[args.robot],
            scene=scene_cfg,
            cameras=cameras,
            # random=args.random,
            render=args.render,
            simulator=args.sim,
            renderer=args.renderer,
            num_envs=args.num_envs,
            headless=args.headless,
        )

    num_envs: int = scenario.num_envs

    if args.sim == "isaacsim":
        scenario.update(decimation=2)
        if scenario.robots[0].name == "franka":
            # use smaller stiffness and damping for fingers for fine-grained control
            from metasim.scenario.robot import BaseActuatorCfg

            scenario.robots[0].actuators["panda_finger_joint1"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )
            scenario.robots[0].actuators["panda_finger_joint2"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")
    traj_filepath = env.traj_filepath
    all_states = None

    ## Data
    tic = time.time()
    if args.action_source == "traj":
        assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
        init_states, all_actions, all_states = get_traj(
            traj_filepath, scenario.robots[0], env.handler
        )  # XXX: only support one robot
        log.info(f"Loaded trajectory joint actions from: {traj_filepath}")
    elif args.action_source == "action_md":
        parsed_actions = load_actions_from_markdown(args.action_md_path)
        # Replay all envs with the same recorded EE sequence by default.
        all_actions = [parsed_actions for _ in range(num_envs)]
        init_states = None
        log.info(f"Loaded {len(parsed_actions)} EE actions from: {args.action_md_path}")
    else:
        if not args.action_json_path:
            raise ValueError("--action-json-path must be provided when --action-source=action_json")
        parsed_actions, source_desc = load_actions_from_json(
            args.action_json_path, episode_index=args.action_json_episode_index
        )
        # Replay all envs with the same recorded EE sequence by default.
        all_actions = [parsed_actions for _ in range(num_envs)]
        init_states = None
        log.info(
            f"Loaded {len(parsed_actions)} EE actions from JSON ({source_desc}): {args.action_json_path}"
        )
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ik_converter = None
    if args.action_source in ("action_md", "action_json"):
        ik_converter = EEActionIKConverter(
            robot_cfg=scenario.robots[0],
            handler=env.handler,
            device=device,
            solver=args.ik_solver,
            action_coordinate_frame=args.ee_action_coordinate_frame,
        )

    ########################################################
    ## Main
    ########################################################

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path)
    os.makedirs("test_output", exist_ok=True)

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    obs_saver.add(obs)

    ## Main loop
    step = 0
    while True:
        log.debug(f"Step {step}")
        tic = time.time()
        if args.object_states:
            ## TODO: merge states replay into env.step function
            if all_states is None:
                raise ValueError("All states are None, please check the trajectory file")
            states = get_states(all_states, step, num_envs)
            env.handler.set_states(states)
            env.handler.refresh_render()
            obs = env.handler.get_states()

            ## XXX: hack
            success = env.checker.check(env.handler, obs)
            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
            if success.all():
                break

        else:
            actions = get_actions(all_actions, step, num_envs, scenario.robots[0])
            if args.action_source in ("action_md", "action_json"):
                action_tensor = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
                actions = ik_converter.convert(action_tensor, obs)
            obs, reward, success, time_out, extras = env.step(actions)

            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")

            if time_out.any():
                log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")

            if success.all() or time_out.all():
                break

        toc = time.time()
        log.trace(f"Time to step: {toc - tic:.2f}s")

        tic = time.time()
        obs_saver.add(obs)
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")
        step += 1

        if args.stop_on_runout and get_runout(all_actions, step):
            log.info("Run out of actions, stopping")
            break

    obs_saver.save()
    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()


if __name__ == "__main__":
    main()
