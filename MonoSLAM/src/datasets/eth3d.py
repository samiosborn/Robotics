# src/datasets/eth3d.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.checks import check_dir
from datasets.image_sequence import ImageSequence, FrameRecord, list_images, build_records_from_files


# ETH3D adapter
@dataclass(frozen=True)
class Eth3DSequenceSpec:
    dataset_root: Path
    seq: str
    rgb_dir: Path

# Find best ETH3D RGB directory
def _find_best_rgb_dir(seq_root: Path) -> Path:
    seq_root = check_dir(seq_root, name="seq_root")

    candidates = []
    for d in seq_root.rglob("rgb"):
        if not d.is_dir():
            continue
        imgs = list_images(d, recursive=False)
        if len(imgs) == 0:
            continue
        candidates.append((len(imgs), d))

    if len(candidates) == 0:
        raise FileNotFoundError(f"Could not find an ETH3D rgb/ folder under: {seq_root}")

    candidates.sort(key=lambda x: (-x[0], x[1].as_posix()))
    return candidates[0][1]

# Resolve ETH3D sequence path
def resolve_eth3d_sequence(dataset_root: Path, seq: str) -> Eth3DSequenceSpec:
    root = check_dir(dataset_root, name="dataset_root")

    s = str(seq).strip()
    if s == "":
        raise ValueError("seq must be a non-empty string")

    seq_root = check_dir(root / s, name="sequence path")

    rgb_dir = _find_best_rgb_dir(seq_root)

    return Eth3DSequenceSpec(dataset_root=root, seq=s, rgb_dir=rgb_dir)

# Load ETH3D sequence
def load_eth3d_sequence(
    dataset_root: Path,
    seq: str,
    *,
    normalise_01: bool = True,
    dtype=None,
    require_timestamps: bool = True,
) -> ImageSequence:
    spec = resolve_eth3d_sequence(dataset_root, seq)

    files = list_images(spec.rgb_dir, recursive=False)

    records: list[FrameRecord] = build_records_from_files(
        files,
        require_timestamps=bool(require_timestamps),
    )

    name = f"eth3d:{spec.seq}"
    seq_obj = ImageSequence(records, name=name, normalise_01=bool(normalise_01), dtype=dtype if dtype is not None else __import__("numpy").float64)
    return seq_obj
