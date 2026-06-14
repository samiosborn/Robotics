# src/datasets/loader.py
from __future__ import annotations

from pathlib import Path

from datasets.image_sequence import ImageSequence


# Load a named dataset sequence by explicit dispatch
def load_sequence(
    dataset_name: str,
    dataset_root: Path,
    seq: str,
    *,
    normalise_01: bool = True,
    dtype=None,
    require_timestamps: bool = True,
) -> ImageSequence:
    name = str(dataset_name).strip().lower()

    if name == "eth3d":
        from datasets.eth3d import load_eth3d_sequence
        return load_eth3d_sequence(
            dataset_root,
            seq,
            normalise_01=normalise_01,
            dtype=dtype,
            require_timestamps=require_timestamps,
        )

    raise ValueError(f"Unsupported dataset name: {dataset_name!r}")
