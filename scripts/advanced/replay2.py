from __future__ import annotations

import logging
import os
import json
import sys
import time
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
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "kitchen_open_drawer_put_bowl"
    robot: str = "franka"
    scene: str | None = None
    # Native LIBERO demos are rendered without raytracing; this avoids overly bright scenes.
    render: RenderCfg = RenderCfg(mode="rasterization")
    # random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 50
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "test_output/tmp"
    save_video_path: str | None = "test_output/test_replay.mp4"
    # Structured dataset export root. If set, replay saves per-env/per-camera frames and metadata.
    save_dataset_dir: str | None = "test_output/collected"
    # Number of replay steps to write into JSONL.
    # -1 means record all steps; otherwise only steps in [0, json_record_steps).
    json_record_steps: int = 10
    # Whether to additionally export per-camera PNG frames for debugging camera pose.
    save_png_frames: bool = False
    # Whether to export legacy JSONL (one record per line).
    save_jsonl: bool = False
    # Additionally export a human-readable JSON array file with indentation.
    save_pretty_json: bool = True
    pretty_json_indent: int = 2
    stop_on_runout: bool = False
    # Optional override for task default trajectory path.
    traj_path: str | None = None

    # Camera controls for replay visualization.
    camera_preset: Literal["default", "libero_front"] = "default"
    camera_pos: tuple[float, float, float] | None = None
    camera_look_at: tuple[float, float, float] | None = None

    # Optional wrist camera injection from an external pose json file.
    # Supported spec keys include:
    # - world camera: {"pos": [...], "look_at": [...]}.
    # - mounted camera: {"mount_pos": [...], "mount_quat": [w, x, y, z], "mount_link": "..."}.
    # The json can be either a single spec object, or a dict keyed by task/robot/default.
    enable_wrist_camera: bool = True
    wrist_camera_pose_path: str | None = None
    wrist_camera_name: str = "robot0_eye_in_hand"
    wrist_camera_width: int = 256
    wrist_camera_height: int = 256
    wrist_camera_focal_length: float = 24.0
    wrist_camera_mount_to: str | None = "franka"
    wrist_camera_mount_link: str | None = "panda_hand"
    wrist_camera_mount_pos: tuple[float, float, float] | None = (0.11, 0.0, -0.06)
    wrist_camera_mount_quat: tuple[float, float, float, float] | None = (-0.1, -0.7, -0.7, -0.1)

    # Optional replay timing alignment.
    # If set, replay will target this control frequency by adjusting decimation.
    control_freq: float | None = 20.0
    # Optional explicit physics timestep override.
    sim_dt: float | None = None
    # Optional explicit decimation override (wins over control_freq).
    decimation: int | None = None

    # Optional gripper close stabilization for migrated trajectories.
    # When enabled, any finger target <= threshold is replaced with close_value.
    gripper_close_threshold: float | None = None
    gripper_close_value: float = 0.004

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


def _to_numpy_uint8(rgb_tensor: torch.Tensor) -> NDArray:
    """Convert camera RGB tensor to uint8 HWC image."""
    img = rgb_tensor.detach().cpu().numpy()
    if img.dtype != np.uint8:
        # Accept [0, 1] float or already [0, 255] float.
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0.0, 255.0)
        img = img.astype(np.uint8)
    return img


def _jsonable(obj):
    """Best-effort conversion for JSON logging."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def _to_float_tuple(raw, expected_len: int, key: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_len:
        raise ValueError(f"Invalid `{key}`: expected length {expected_len}, got {raw}")
    return tuple(float(v) for v in raw)


def _is_camera_pose_spec(data) -> bool:
    if not isinstance(data, dict):
        return False
    keys = {
        "pos",
        "look_at",
        "mount_pos",
        "mount_quat",
        "mount_link",
        "mount_to",
        "name",
        "width",
        "height",
        "focal_length",
    }
    return any(k in data for k in keys)


def _load_wrist_pose_spec(path: str, task: str, robot: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if _is_camera_pose_spec(payload):
        return payload

    if not isinstance(payload, dict):
        raise ValueError("wrist camera pose json must be an object")

    if task in payload and _is_camera_pose_spec(payload[task]):
        return payload[task]
    if robot in payload and _is_camera_pose_spec(payload[robot]):
        return payload[robot]
    if "default" in payload and _is_camera_pose_spec(payload["default"]):
        return payload["default"]

    raise ValueError(
        "Unable to find wrist camera spec in json. Expected a direct spec or one of keys: "
        f"`{task}`, `{robot}`, `default`."
    )


def _get_default_mount_link(task_cls) -> str | None:
    try:
        robots = getattr(task_cls.scenario, "robots", None)
        if robots and len(robots) > 0:
            return getattr(robots[0], "ee_body_name", None)
    except Exception:
        return None
    return None


def _build_wrist_camera_cfg(task_cls) -> PinholeCameraCfg | None:
    if not args.enable_wrist_camera:
        return None

    if args.wrist_camera_pose_path is not None:
        spec = _load_wrist_pose_spec(args.wrist_camera_pose_path, args.task, args.robot)
    else:
        # Built-in mounted wrist camera defaults for Franka.
        spec = {
            "name": args.wrist_camera_name,
            "width": args.wrist_camera_width,
            "height": args.wrist_camera_height,
            "focal_length": args.wrist_camera_focal_length,
            "mount_to": args.wrist_camera_mount_to,
            "mount_link": args.wrist_camera_mount_link,
            "mount_pos": args.wrist_camera_mount_pos,
            "mount_quat": args.wrist_camera_mount_quat,
        }

    cam_name = str(spec.get("name", args.wrist_camera_name))
    width = int(spec.get("width", args.wrist_camera_width))
    height = int(spec.get("height", args.wrist_camera_height))
    focal_length = float(spec.get("focal_length", args.wrist_camera_focal_length))

    mount_pos_raw = spec.get("mount_pos")
    mount_quat_raw = spec.get("mount_quat")
    mount_link = spec.get("mount_link", args.wrist_camera_mount_link)
    mount_to = spec.get("mount_to", args.wrist_camera_mount_to)

    if mount_to is None and args.robot != "None":
        mount_to = args.robot

    if mount_link is None:
        mount_link = _get_default_mount_link(task_cls)

    if mount_pos_raw is not None or mount_quat_raw is not None:
        if mount_pos_raw is None or mount_quat_raw is None:
            raise ValueError("Mounted wrist camera requires both `mount_pos` and `mount_quat`.")
        mount_pos = _to_float_tuple(mount_pos_raw, 3, "mount_pos")
        mount_quat = _to_float_tuple(mount_quat_raw, 4, "mount_quat")
        if mount_to is None or mount_link is None:
            raise ValueError(
                "Mounted wrist camera requires `mount_to` and `mount_link` "
                "(either in pose json or cli defaults)."
            )
        return PinholeCameraCfg(
            name=cam_name,
            width=width,
            height=height,
            focal_length=focal_length,
            # Mounted camera uses mount pose; pos/look_at values are fallback placeholders.
            pos=(0.0, 0.0, 1.0),
            look_at=(0.0, 0.0, 0.0),
            mount_to=str(mount_to),
            mount_link=str(mount_link),
            mount_pos=mount_pos,
            mount_quat=mount_quat,
        )

    pos = spec.get("pos")
    look_at = spec.get("look_at")
    if pos is None or look_at is None:
        raise ValueError(
            "Wrist camera pose json must provide either mounted pose "
            "(`mount_pos` + `mount_quat`) or world pose (`pos` + `look_at`)."
        )
    return PinholeCameraCfg(
        name=cam_name,
        width=width,
        height=height,
        focal_length=focal_length,
        pos=_to_float_tuple(pos, 3, "pos"),
        look_at=_to_float_tuple(look_at, 3, "look_at"),
    )


class ReplayCollector:
    """Collect per-env/per-camera frames and replay metadata."""

    def __init__(
        self,
        task_name: str,
        num_envs: int,
        dataset_root: str | None,
        image_dir: str | None,
        video_path: str | None,
        main_camera_name: str,
        wrist_camera_name: str,
        json_record_steps: int,
        save_png_frames: bool,
        save_jsonl: bool,
        save_pretty_json: bool,
        pretty_json_indent: int,
    ):
        self.task_name = task_name
        self.num_envs = num_envs
        self.obs_saver = ObsSaver(image_dir=image_dir, video_path=video_path)
        self.records: list[dict] = []
        self.dataset_dir: Path | None = None
        self.frame_root: Path | None = None
        self.meta_path: Path | None = None
        self.main_camera_name = main_camera_name
        self.wrist_camera_name = wrist_camera_name
        self.save_png_frames = save_png_frames
        self.save_jsonl = save_jsonl
        self.max_json_steps = None if json_record_steps < 0 else int(json_record_steps)
        self.save_pretty_json = save_pretty_json
        self.pretty_json_indent = max(int(pretty_json_indent), 0)
        self.pretty_meta_path: Path | None = None

        if dataset_root is not None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.dataset_dir = Path(dataset_root) / task_name / stamp
            # Always create dataset directory so JSON/JSONL can be written even without PNG frames.
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            if self.save_png_frames:
                self.frame_root = self.dataset_dir / "frames"
                self.frame_root.mkdir(parents=True, exist_ok=True)
            if self.save_jsonl:
                self.meta_path = self.dataset_dir / "trajectory.jsonl"
            if self.save_pretty_json:
                self.pretty_meta_path = self.dataset_dir / "trajectory.json"
            log.info(f"Structured replay dataset dir: {self.dataset_dir}")
        elif self.save_png_frames:
            log.warning("`save_png_frames=True` but `save_dataset_dir` is None; PNG export is disabled.")

    def _pick_two_images(self, camera_images: dict[str, NDArray]) -> tuple[NDArray | None, NDArray | None]:
        wrist_image = camera_images.get(self.wrist_camera_name)

        # Prefer explicitly configured main camera; otherwise fallback to any non-wrist camera.
        image = camera_images.get(self.main_camera_name)
        if image is None:
            for cam_name, cam_img in camera_images.items():
                if cam_name != self.wrist_camera_name:
                    image = cam_img
                    break
        if image is None and camera_images:
            image = next(iter(camera_images.values()))

        return image, wrist_image

    def add(
        self,
        state: TensorState,
        step: int,
        actions=None,
        reward=None,
        success=None,
        time_out=None,
    ):
        # Keep existing combined output behavior.
        self.obs_saver.add(state)

        if self.dataset_dir is None:
            return

        should_record_json = self.max_json_steps is None or step < self.max_json_steps

        if not state.cameras:
            log.warning("No camera found in state.cameras, skip structured frame dump.")
            return

        camera_payload: dict[str, torch.Tensor] = {
            cam_name: cam_state.rgb for cam_name, cam_state in state.cameras.items()
        }

        # Write one metadata record per env per step.
        for env_id in range(self.num_envs):
            camera_images: dict[str, NDArray] = {}
            for cam_name, rgb_batch in camera_payload.items():
                if env_id >= rgb_batch.shape[0]:
                    continue
                img = _to_numpy_uint8(rgb_batch[env_id])
                camera_images[cam_name] = img
                if self.save_png_frames and self.frame_root is not None:
                    cam_dir = self.frame_root / f"env_{env_id:03d}" / cam_name
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = cam_dir / f"frame_{step:06d}.png"
                    iio.imwrite(frame_path, img)

            image, wrist_image = self._pick_two_images(camera_images)

            if should_record_json:
                rec = {
                    # Keep key names aligned with SFT samples and store image pixels directly.
                    "prompt": self.task_name,
                    # Use env_id as episode index because get_actions/get_states slice demos by env index.
                    "episode_index": int(env_id),
                    "timestep": int(step),
                    "image": _jsonable(image),
                    "wrist_image": _jsonable(wrist_image),
                    # Placeholders to be filled by your downstream post-processing step.
                    "state": [],
                    "actions": [],
                }
                self.records.append(rec)

    def save(self):
        self.obs_saver.save()
        if self.meta_path is not None:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                for rec in self.records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.info(f"Saved {len(self.records)} metadata rows to {self.meta_path}")

        if self.pretty_meta_path is not None:
            self._save_readable_json(self.pretty_meta_path)
            log.info(f"Saved pretty JSON to {self.pretty_meta_path}")

    @staticmethod
    def _compact_json(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _save_readable_json(self, output_path: Path):
        """Write a readable JSON array without exploding pixel arrays over thousands of lines."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            total = len(self.records)
            for idx, rec in enumerate(self.records):
                prompt = rec.get("prompt", "")
                image = rec.get("image", [])
                wrist_image = rec.get("wrist_image", [])
                state = rec.get("state", [])
                actions = rec.get("actions", [])

                f.write("  {\n")
                f.write(f"    \"prompt\": {json.dumps(prompt, ensure_ascii=False)},\n")
                f.write(f"    \"image\": {self._compact_json(image)},\n")
                f.write(f"    \"wrist_image\": {self._compact_json(wrist_image)},\n")
                f.write(f"    \"state\": {self._compact_json(state)},\n")
                f.write(f"    \"actions\": {self._compact_json(actions)}\n")
                f.write("  }")
                if idx < total - 1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")


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

    cam_focal_length = 22

    if args.camera_pos is not None and args.camera_look_at is not None:
        cam_pos = args.camera_pos
        cam_look_at = args.camera_look_at
    elif args.camera_preset == "libero_front" or args.task.startswith("libero_"):
        # Use a slightly tighter framing on the tabletop workspace.
        cam_pos = (1.0, 0.0, 1.22)
        cam_look_at = (-0.02, 0.0, 0.64)
        cam_focal_length = 40.0
    else:
        cam_pos = (0.55, 0.0, 1.4)
        cam_look_at = (-0.28, 0.0, 0.77)

    camera = PinholeCameraCfg(pos=cam_pos, look_at=cam_look_at, focal_length=cam_focal_length)
    cameras = list(task_cls.scenario.cameras) if use_task_cameras else [camera]

    wrist_camera = _build_wrist_camera_cfg(task_cls)
    if wrist_camera is not None:
        cameras.append(wrist_camera)
        log.info(f"Injected wrist camera: name={wrist_camera.name}")

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

    # if args.sim_dt is not None:
    #     scenario.sim_params.dt = float(args.sim_dt)

    # # Prefer explicit decimation override; otherwise infer from requested control frequency.
    # if args.decimation is not None:
    #     scenario.update(decimation=max(int(args.decimation), 1))
    # elif args.control_freq is not None and args.control_freq > 0:
    #     physics_dt = get_effective_physics_dt(args.sim, scenario.sim_params.dt)
    #     target_decimation = max(int(round((1.0 / args.control_freq) / physics_dt)), 1)
    #     scenario.update(decimation=target_decimation)

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

    # physics_dt = get_effective_physics_dt(args.sim, scenario.sim_params.dt)
    # effective_control_hz = 1.0 / (physics_dt * scenario.decimation)
    # log.info(
    #     f"Replay timing: sim={args.sim}, physics_dt={physics_dt:.6f}s, decimation={scenario.decimation}, "
    #     f"control_hz={effective_control_hz:.2f}"
    # )

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")
    traj_filepath = args.traj_path if args.traj_path is not None else env.traj_filepath
    log.info(f"Using trajectory file: {traj_filepath}")
    ## Data
    tic = time.time()
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(
        traj_filepath, scenario.robots[0], env.handler
    )  # XXX: only support one robot
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ########################################################
    ## Main
    ########################################################

    collector = ReplayCollector(
        task_name=args.task,
        num_envs=num_envs,
        dataset_root=args.save_dataset_dir,
        image_dir=args.save_image_dir,
        video_path=args.save_video_path,
        main_camera_name="camera0",
        wrist_camera_name=args.wrist_camera_name,
        json_record_steps=args.json_record_steps,
        save_png_frames=args.save_png_frames,
        save_jsonl=args.save_jsonl,
        save_pretty_json=args.save_pretty_json,
        pretty_json_indent=args.pretty_json_indent,
    )
    os.makedirs("test_output", exist_ok=True)

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    collector.add(obs, step=0)

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
            # if args.gripper_close_threshold is not None:
            #     apply_gripper_close_floor(
            #         actions,
            #         robot_name=scenario.robots[0].name,
            #         threshold=float(args.gripper_close_threshold),
            #         close_value=float(args.gripper_close_value),
            #     )
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
        collector.add(
            obs,
            step=step + 1,
            actions=actions,
            reward=reward if not args.object_states else None,
            success=success if not args.object_states else None,
            time_out=time_out if not args.object_states else None,
        )
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")
        step += 1

        if args.stop_on_runout and get_runout(all_actions, step):
            log.info("Run out of actions, stopping")
            break

    collector.save()
    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()


if __name__ == "__main__":
    main()
