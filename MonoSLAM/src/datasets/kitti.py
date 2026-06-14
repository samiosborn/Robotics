# src/datasets/kitti.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.checks import check_dir
from datasets.image_sequence import FrameRecord, ImageSequence


# Resolve the image_0 directory for a given sequence
def _image0_dir(dataset_root: Path, seq: str) -> Path:
    d = dataset_root / "sequences" / seq / "image_0"
    return check_dir(d, name="kitti image_0")


# Collect PNG files sorted by numeric stem
def _sorted_pngs(image_dir: Path) -> list[Path]:
    files = [p for p in image_dir.glob("*.png") if p.is_file()]
    try:
        files.sort(key=lambda p: int(p.stem))
    except ValueError:
        files.sort(key=lambda p: p.stem)
    return files


# Build deterministic FrameRecords from a sorted file list
# Frame index is used as the timestamp
def _build_kitti_records(files: list[Path]) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for idx, p in enumerate(files):
        records.append(FrameRecord(frame_id=p.stem, timestamp=float(idx), path=p))
    return records


# Load KITTI odometry monocular grayscale left camera sequence
def load_kitti_sequence(
    dataset_root: Path,
    seq: str,
    *,
    normalise_01: bool = True,
    dtype=None,
) -> ImageSequence:
    root = check_dir(dataset_root, name="dataset_root")
    s = str(seq).strip()
    if s == "":
        raise ValueError("seq must be a non-empty string")

    image_dir = _image0_dir(root, s)
    files = _sorted_pngs(image_dir)

    if len(files) == 0:
        raise FileNotFoundError(f"No PNG files found in: {image_dir}")

    records = _build_kitti_records(files)
    name = f"kitti:{s}"
    _dtype = dtype if dtype is not None else np.float64
    return ImageSequence(records, name=name, normalise_01=bool(normalise_01), dtype=_dtype)
