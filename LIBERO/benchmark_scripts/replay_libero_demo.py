import argparse
import os

import cv2
import h5py
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Optional task name to resolve task_id automatically (overrides --task_id).",
    )
    parser.add_argument("--demo_id", type=int, default=0)
    parser.add_argument(
        "--demo_file",
        type=str,
        default=None,
        help="Optional absolute path to a specific hdf5 demo file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional override for the LIBERO dataset root directory.",
    )
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["video", "gif"],
        default="video",
        help="Output a compressed video file or a GIF for easier preview in VS Code.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_videos",
        help="Directory for saving replay videos.",
    )
    return parser.parse_args()


def resolve_task_id(benchmark_instance, task_name: str) -> int:
    """Resolve task_id from a task name or unique substring."""
    query = task_name.strip().lower()
    exact_matches = []
    partial_matches = []

    for idx in range(benchmark_instance.n_tasks):
        task = benchmark_instance.get_task(idx)
        name = task.name.lower()
        if name == query:
            exact_matches.append((idx, task.name))
        elif query in name:
            partial_matches.append((idx, task.name))

    if len(exact_matches) == 1:
        return exact_matches[0][0]

    if len(exact_matches) > 1:
        choices = ", ".join([f"{idx}:{name}" for idx, name in exact_matches])
        raise ValueError(f"Multiple exact matches for task_name='{task_name}': {choices}")

    if len(partial_matches) == 1:
        return partial_matches[0][0]

    if len(partial_matches) > 1:
        choices = ", ".join([f"{idx}:{name}" for idx, name in partial_matches[:10]])
        raise ValueError(
            f"task_name='{task_name}' matches multiple tasks. Please be more specific. Matches: {choices}"
        )

    raise ValueError(f"No task matches task_name='{task_name}' in benchmark '{benchmark_instance.name}'.")


def to_bgr_frame(rgb_frame):
    return rgb_frame[::-1, :, ::-1].copy()


def create_video_writer(output_path, fps, frame_size):
    codecs = [
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
    ]

    base_path, _ext = os.path.splitext(output_path)
    last_error = None
    for codec, suffix in codecs:
        candidate_path = base_path + suffix
        writer = cv2.VideoWriter(
            candidate_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, candidate_path, codec
        last_error = codec

    raise RuntimeError(
        f"Failed to create a video writer. Last attempted codec: {last_error}"
    )


def to_pil_frame(bgr_frame):
    return Image.fromarray(bgr_frame[:, :, ::-1])


def main():
    args = parse_args()

    benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark_name]()
    task_id = args.task_id
    if args.task_name is not None:
        task_id = resolve_task_id(benchmark_instance, args.task_name)

    task = benchmark_instance.get_task(task_id)

    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    if args.demo_file is not None:
        demo_file = args.demo_file
    else:
        dataset_root = args.dataset_root or get_libero_path("datasets")
        demo_file = os.path.join(
            dataset_root, benchmark_instance.get_task_demonstration(task_id)
        )

    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"Demo file does not exist: {demo_file}")

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_basename = (
        f"{args.benchmark_name}_task{task_id}_demo{args.demo_id}_{task.name}"
    )
    output_path = os.path.join(args.output_dir, output_basename + ".mp4")

    env = OffScreenRenderEnv(**env_args)
    env.reset()

    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{args.demo_id}"
        if demo_key not in f:
            raise KeyError(f"Demo {demo_key} not found in {demo_file}")

        demo_group = f[demo_key]
        actions = demo_group["actions"][()]
        states = demo_group["states"][()]

    obs = env.set_init_state(states[0])
    first_frame = to_bgr_frame(obs["agentview_image"])
    frames = [first_frame]

    writer = None
    codec = None
    if args.output_format == "video":
        writer, output_path, codec = create_video_writer(
            output_path,
            args.fps,
            (first_frame.shape[1], first_frame.shape[0]),
        )
        writer.write(first_frame)

    num_steps = len(actions) if args.max_steps < 0 else min(len(actions), args.max_steps)
    for step_idx in range(num_steps):
        obs, _reward, _done, _info = env.step(actions[step_idx].tolist())
        frame = to_bgr_frame(obs["agentview_image"])
        frames.append(frame)
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    else:
        output_path = os.path.join(args.output_dir, output_basename + ".gif")
        pil_frames = [to_pil_frame(frame) for frame in frames]
        duration_ms = max(int(1000 / max(args.fps, 1)), 1)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
    env.close()

    print(f"Saved replay video to {output_path}")
    if codec is not None:
        print(f"Video codec: {codec}")
    print(f"Task: {task.language}")
    print(f"Task id: {task_id}")
    print(f"Demo file: {demo_file}")


if __name__ == "__main__":
    main()