from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np


def load_records(path: Path) -> list[dict[str, Any]]:
	if path.suffix.lower() == ".jsonl":
		records: list[dict[str, Any]] = []
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				obj = json.loads(line)
				if isinstance(obj, dict):
					records.append(obj)
		return records

	with open(path, "r", encoding="utf-8") as f:
		payload = json.load(f)

	if isinstance(payload, list):
		return [x for x in payload if isinstance(x, dict)]
	if isinstance(payload, dict):
		return [payload]
	return []


def to_uint8_image(data: Any) -> np.ndarray | None:
	if data is None:
		return None

	arr = np.asarray(data)
	if arr.size == 0:
		return None
	if arr.ndim != 3 or arr.shape[-1] != 3:
		return None

	if arr.dtype != np.uint8:
		if np.issubdtype(arr.dtype, np.floating):
			vmax = float(np.max(arr)) if arr.size > 0 else 0.0
			if vmax <= 1.0:
				arr = arr * 255.0
			arr = np.clip(arr, 0.0, 255.0)
		else:
			arr = np.clip(arr, 0, 255)
		arr = arr.astype(np.uint8)
	return arr


def make_pair_image(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
	if left is None and right is None:
		return None

	if left is None:
		h, w = right.shape[:2]
		left = np.zeros((h, w, 3), dtype=np.uint8)
	if right is None:
		h, w = left.shape[:2]
		right = np.zeros((h, w, 3), dtype=np.uint8)

	if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
		h = max(left.shape[0], right.shape[0])
		w = max(left.shape[1], right.shape[1])
		canvas_left = np.zeros((h, w, 3), dtype=np.uint8)
		canvas_right = np.zeros((h, w, 3), dtype=np.uint8)
		canvas_left[: left.shape[0], : left.shape[1]] = left
		canvas_right[: right.shape[0], : right.shape[1]] = right
		left, right = canvas_left, canvas_right

	return np.concatenate([left, right], axis=1)


def render_records(
	records: list[dict[str, Any]],
	output_dir: Path,
	start_index: int,
	max_samples: int,
) -> int:
	output_dir.mkdir(parents=True, exist_ok=True)

	rendered = 0
	end = len(records) if max_samples < 0 else min(len(records), start_index + max_samples)

	for idx in range(start_index, end):
		rec = records[idx]
		main = to_uint8_image(rec.get("image"))
		wrist = to_uint8_image(rec.get("wrist_image"))
		pair = make_pair_image(main, wrist)

		if main is not None:
			iio.imwrite(output_dir / f"sample_{idx:06d}_main.png", main)
		if wrist is not None:
			iio.imwrite(output_dir / f"sample_{idx:06d}_wrist.png", wrist)
		if pair is not None:
			iio.imwrite(output_dir / f"sample_{idx:06d}_pair.png", pair)

		rendered += 1

	return rendered


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Render image/wrist_image arrays from trajectory json to PNG files.")
	parser.add_argument("json_path", type=str, help="Path to trajectory.json or trajectory.jsonl")
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Output folder for rendered PNG files. Defaults to <json_dir>/preview_<json_name>",
	)
	parser.add_argument("--start-index", type=int, default=0, help="Start sample index")
	parser.add_argument(
		"--max-samples",
		type=int,
		default=20,
		help="How many samples to render. Use -1 to render all.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	json_path = Path(args.json_path).expanduser().resolve()

	if not json_path.exists():
		raise FileNotFoundError(f"Input file not found: {json_path}")

	records = load_records(json_path)
	if not records:
		raise ValueError(f"No valid records found in: {json_path}")

	if args.output_dir is None:
		default_dir_name = f"preview_{json_path.stem}"
		output_dir = json_path.parent / default_dir_name
	else:
		output_dir = Path(args.output_dir).expanduser().resolve()

	rendered = render_records(
		records=records,
		output_dir=output_dir,
		start_index=max(0, int(args.start_index)),
		max_samples=int(args.max_samples),
	)

	print(f"Loaded records: {len(records)}")
	print(f"Rendered samples: {rendered}")
	print(f"Output dir: {output_dir}")


if __name__ == "__main__":
	main()
