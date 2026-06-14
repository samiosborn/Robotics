# tests/datasets/test_kitti.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# Build a minimal synthetic KITTI image_0 directory
def _make_image0(tmp_path: Path, stems: list[str]) -> Path:
    image_dir = tmp_path / "sequences" / "00" / "image_0"
    image_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        img = Image.fromarray(np.zeros((4, 6), dtype=np.uint8), mode="L")
        img.save(str(image_dir / f"{stem}.png"))
    return tmp_path


# Records are ordered by numeric stem ascending
def test_records_numeric_order(tmp_path):
    from datasets.kitti import load_kitti_sequence

    _make_image0(tmp_path, ["000002", "000000", "000001"])
    seq = load_kitti_sequence(tmp_path, "00")

    ids = [r.frame_id for r in seq.records]
    assert ids == ["000000", "000001", "000002"]


# Frame index is used as timestamp, not the stem value
def test_timestamps_are_frame_indices(tmp_path):
    from datasets.kitti import load_kitti_sequence

    _make_image0(tmp_path, ["000010", "000020", "000030"])
    seq = load_kitti_sequence(tmp_path, "00")

    timestamps = [r.timestamp for r in seq.records]
    assert timestamps == [0.0, 1.0, 2.0]


# frame_id is the numeric stem string
def test_frame_id_is_stem(tmp_path):
    from datasets.kitti import load_kitti_sequence

    _make_image0(tmp_path, ["000000", "000001"])
    seq = load_kitti_sequence(tmp_path, "00")

    assert seq.records[0].frame_id == "000000"
    assert seq.records[1].frame_id == "000001"


# Sequence name encodes the dataset and sequence identifier
def test_sequence_name(tmp_path):
    from datasets.kitti import load_kitti_sequence

    _make_image0(tmp_path, ["000000"])
    seq = load_kitti_sequence(tmp_path, "00")

    assert seq.name == "kitti:00"


# Sequence length matches number of PNG files
def test_sequence_length(tmp_path):
    from datasets.kitti import load_kitti_sequence

    _make_image0(tmp_path, ["000000", "000001", "000002"])
    seq = load_kitti_sequence(tmp_path, "00")

    assert len(seq) == 3


# Missing image_0 directory raises an informative error
def test_missing_image0_raises(tmp_path):
    from datasets.kitti import load_kitti_sequence

    with pytest.raises(Exception):
        load_kitti_sequence(tmp_path, "00")


# Empty image_0 directory raises FileNotFoundError
def test_empty_image0_raises(tmp_path):
    from datasets.kitti import load_kitti_sequence

    image_dir = tmp_path / "sequences" / "00" / "image_0"
    image_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        load_kitti_sequence(tmp_path, "00")


# Loader dispatch for "kitti" name routes to KITTI adapter
def test_loader_dispatch_kitti(tmp_path):
    from datasets.loader import load_sequence

    _make_image0(tmp_path, ["000000", "000001"])
    seq = load_sequence("kitti", tmp_path, "00")

    assert seq.name == "kitti:00"
    assert len(seq) == 2


# Loader dispatch for unsupported name still raises
def test_loader_dispatch_unsupported_raises(tmp_path):
    from datasets.loader import load_sequence

    with pytest.raises(ValueError, match="Unsupported dataset name"):
        load_sequence("unknown_dataset", tmp_path, "00")


# get() returns a valid greyscale array with expected shape
def test_get_returns_array(tmp_path):
    from datasets.kitti import load_kitti_sequence

    _make_image0(tmp_path, ["000000"])
    seq = load_kitti_sequence(tmp_path, "00")

    arr, ts, frame_id = seq.get(0)
    assert arr.ndim == 2
    assert arr.shape == (4, 6)
    assert float(ts) == 0.0
    assert frame_id == "000000"
