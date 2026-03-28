from __future__ import annotations

import copy
import logging
import os
import time
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

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.robot import RobotCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "libero_10.living_room_scene1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg(mode="rasterization")

    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    num_envs: int = 1
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    save_image_dir: str | None = "test_output/tmp"
    save_video_path: str | None = "test_output/test_replay_pd_gripper.mp4"
    stop_on_runout: bool = False
    traj_path: str | None = None

    camera_preset: Literal["default", "libero_front"] = "default"
    camera_pos: tuple[float, float, float] | None = None
    camera_look_at: tuple[float, float, float] | None = None

    # Gripper controller params (position-target shaping before env.step)
    enable_gripper_controller: bool = True
    gripper_joint1: str = "panda_finger_joint1"
    gripper_joint2: str = "panda_finger_joint2"
    gripper_open_q: float = 0.04
    gripper_close_q: float = 0.004
    gripper_cmd_threshold: float = 0.02
    gripper_rate_limit: float = 0.006

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


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
    return all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])


class ObsSaver:
    """Save observations to images and/or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []
        self.image_idx = 0

    def add(self, state: TensorState):
        if self.image_dir is None and self.video_path is None:
            return
        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


class GripperReplayController:
    """
    Smooth and stabilize gripper targets for replay.

    This does not change task selection or replay flow from replay2.py.
    It only post-processes finger position targets before env.step(actions).
    """

    def __init__(
        self,
        num_envs: int,
        open_q: float,
        close_q: float,
        cmd_threshold: float,
        rate_limit: float,
        finger_keys: tuple[str, str],
    ):
        self.num_envs = num_envs
        self.open_q = float(open_q)
        self.close_q = float(close_q)
        self.cmd_threshold = float(cmd_threshold)
        self.rate_limit = float(rate_limit)
        self.finger_keys = finger_keys
        self.current_target = [self.open_q for _ in range(num_envs)]

    def _extract_cmd(self, dof_pos_target: dict) -> float | None:
        values = [dof_pos_target.get(k) for k in self.finger_keys if k in dof_pos_target]
        if len(values) == 0:
            return None
        return float(np.mean(values))

    def _step_target(self, env_i: int, desired: float) -> float:
        cur = self.current_target[env_i]
        delta = np.clip(desired - cur, -self.rate_limit, self.rate_limit)
        nxt = float(np.clip(cur + delta, min(self.close_q, self.open_q), max(self.close_q, self.open_q)))
        self.current_target[env_i] = nxt
        return nxt

    def apply(self, actions: list[dict], robot_name: str) -> list[dict]:
        patched = copy.deepcopy(actions)
        for env_i, action in enumerate(patched):
            if not isinstance(action, dict):
                continue
            robot_action = action.get(robot_name)
            if not isinstance(robot_action, dict):
                continue
            dof_pos_target = robot_action.get("dof_pos_target")
            if not isinstance(dof_pos_target, dict):
                continue

            cmd = self._extract_cmd(dof_pos_target)
            if cmd is None:
                continue

            desired = self.close_q if cmd <= self.cmd_threshold else self.open_q
            target_q = self._step_target(env_i, desired)
            dof_pos_target[self.finger_keys[0]] = target_q
            dof_pos_target[self.finger_keys[1]] = target_q
        return patched


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
        cam_pos = (1.0, 0.0, 1.22)
        cam_look_at = (-0.02, 0.0, 0.64)
        cam_focal_length = 40.0
    else:
        cam_pos = (1.5, -1.5, 1.5)
        cam_look_at = (0.0, 0.0, 0.0)

    camera = PinholeCameraCfg(pos=cam_pos, look_at=cam_look_at, focal_length=cam_focal_length)
    cameras = task_cls.scenario.cameras if use_task_cameras else [camera]

    scene_cfg = task_cls.scenario.scene if task_cls.scenario.scene is not None else args.scene
    scenario = task_cls.scenario.update(
        robots=[args.robot] if args.robot != "None" else None,
        scene=scene_cfg,
        cameras=cameras,
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    num_envs: int = scenario.num_envs
    if args.sim == "isaacsim":
        scenario.update(decimation=2)

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    log.trace(f"Time to launch: {time.time() - tic:.2f}s")

    traj_filepath = args.traj_path if args.traj_path is not None else env.traj_filepath
    log.info(f"Using trajectory file: {traj_filepath}")
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."

    tic = time.time()
    init_states, all_actions, all_states = get_traj(traj_filepath, scenario.robots[0], env.handler)
    log.trace(f"Time to load data: {time.time() - tic:.2f}s")
    _ = init_states

    gripper_controller = GripperReplayController(
        num_envs=num_envs,
        open_q=args.gripper_open_q,
        close_q=args.gripper_close_q,
        cmd_threshold=args.gripper_cmd_threshold,
        rate_limit=args.gripper_rate_limit,
        finger_keys=(args.gripper_joint1, args.gripper_joint2),
    )

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path)
    os.makedirs("test_output", exist_ok=True)

    obs, extras = env.reset()
    _ = extras
    obs_saver.add(obs)

    step = 0
    while True:
        if args.object_states:
            if all_states is None:
                raise ValueError("All states are None, please check the trajectory file")
            states = get_states(all_states, step, num_envs)
            env.handler.set_states(states)
            env.handler.refresh_render()
            obs = env.handler.get_states()
            success = env.checker.check(env.handler, obs)
            if success.all():
                break
        else:
            actions = get_actions(all_actions, step, num_envs, scenario.robots[0])
            if args.enable_gripper_controller:
                actions = gripper_controller.apply(actions, robot_name=scenario.robots[0].name)

            obs, reward, success, time_out, extras = env.step(actions)
            _ = reward, extras

            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
            if time_out.any():
                log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")
            if success.all() or time_out.all():
                break

        obs_saver.add(obs)
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
