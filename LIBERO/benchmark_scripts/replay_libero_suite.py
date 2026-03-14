import argparse
import subprocess
import sys

from libero.libero import benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, default="libero_10")
    parser.add_argument("--demo_id", type=int, default=0)
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
    parser.add_argument("--output_dir", type=str, default="benchmark_videos")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark_name]()

    for task_id in range(benchmark_instance.get_num_tasks()):
        cmd = [
            sys.executable,
            "benchmark_scripts/replay_libero_demo.py",
            "--benchmark_name",
            args.benchmark_name,
            "--task_id",
            str(task_id),
            "--demo_id",
            str(args.demo_id),
        ]
        if args.dataset_root is not None:
            cmd.extend([
                "--dataset_root",
                args.dataset_root,
            ])
        cmd.extend([
            "--camera_height",
            str(args.camera_height),
            "--camera_width",
            str(args.camera_width),
            "--fps",
            str(args.fps),
            "--max_steps",
            str(args.max_steps),
            "--output_dir",
            args.output_dir,
        ])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()