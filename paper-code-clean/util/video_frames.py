"""Convert videos to numbered PNG frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mov", ".avi", ".wmv", ".mp4"}


def capture_frames(video_path: Path | str, *, delete_original: bool = False) -> int:
    video_path = Path(video_path)
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    frame_count = 0
    try:
        while True:
            success, image = capture.read()
            if not success:
                break
            frame_count += 1
            output_path = video_path.with_name(f"{video_path.stem}_frame{frame_count}.png")
            if not cv2.imwrite(str(output_path), image):
                raise OSError(f"Could not write frame: {output_path}")
    finally:
        capture.release()

    if delete_original:
        video_path.unlink()
    return frame_count


def capture_tree(root: Path | str, *, delete_original: bool = False) -> int:
    root = Path(root)
    total = 0
    videos = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    for video in videos:
        total += capture_frames(video, delete_original=delete_original)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--delete-original", action="store_true")
    args = parser.parse_args()
    print(f"Wrote {capture_tree(args.root, delete_original=args.delete_original)} frames")


if __name__ == "__main__":
    main()
