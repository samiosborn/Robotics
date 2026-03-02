# src/datasets/image_sequence.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from core.checks import check_dir, check_file

# Frame record
@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    timestamp: float
    path: Path


# Timestamp parsing
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Parse timestamp from stem
def parse_timestamp_from_stem(stem: str) -> float | None:
    s = str(stem).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        pass
    m = _FLOAT_RE.search(s)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


# Greyscale Image loading
def load_greyscale(
    path: Path,
    *,
    normalise_01: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    p = check_file(path, name="path")

    im = Image.open(str(p))
    im = im.convert("L")
    arr = np.asarray(im)

    if not bool(normalise_01):
        return np.asarray(arr)

    out = np.asarray(arr, dtype=dtype) / 255.0
    return out


# Listing image files
def list_images(
    root: Path,
    *,
    patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp"),
    recursive: bool = True,
) -> list[Path]:
    r = check_dir(root, name="root")

    out: list[Path] = []
    for pat in patterns:
        if recursive:
            out.extend(sorted(r.rglob(pat)))
        else:
            out.extend(sorted(r.glob(pat)))

    out = [p for p in out if p.is_file()]
    out.sort(key=lambda p: p.as_posix())
    return out


# Records builder
def build_records_from_files(
    files: Iterable[Path],
    *,
    timestamp_fn: Callable[[Path], float | None] | None = None,
    require_timestamps: bool = False,
) -> list[FrameRecord]:
    fs = [Path(p) for p in files]

    if timestamp_fn is None:

        def _ts(p: Path) -> float | None:
            return parse_timestamp_from_stem(p.stem)

        timestamp_fn = _ts

    tmp: list[tuple[Path, float | None]] = []
    for p in fs:
        tmp.append((p, timestamp_fn(p)))

    have_all = all(t is not None for _, t in tmp)

    if bool(require_timestamps) and not have_all:
        bad = [p.name for p, t in tmp if t is None]
        show = ", ".join(bad[:10])
        raise ValueError(f"missing timestamps for {len(bad)} frames (first: {show})")

    if have_all:
        tmp2 = [(p, float(t)) for (p, t) in tmp]
        tmp2.sort(key=lambda pt: (pt[1], pt[0].as_posix()))
        records: list[FrameRecord] = []
        for p, ts in tmp2:
            records.append(FrameRecord(frame_id=p.stem, timestamp=float(ts), path=p))
        return records

    tmp.sort(key=lambda pt: pt[0].as_posix())
    records = []
    for i, (p, _) in enumerate(tmp):
        records.append(FrameRecord(frame_id=p.stem, timestamp=float(i), path=p))
    return records


# Sequence wrapper
class ImageSequence:
    def __init__(
        self,
        records: list[FrameRecord],
        *,
        name: str = "sequence",
        normalise_01: bool = True,
        dtype=np.float64,
    ):
        self._name = str(name)
        self._records = list(records)
        self._normalise_01 = bool(normalise_01)
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def records(self) -> list[FrameRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return int(len(self._records))

    def frame_info(self, i: int) -> FrameRecord:
        n = len(self._records)
        if not isinstance(i, int):
            raise ValueError("index must be int")
        if i < 0 or i >= n:
            raise IndexError(f"index out of range: {i} (n={n})")
        return self._records[i]

    def get(self, i: int) -> tuple[np.ndarray, float, str]:
        rec = self.frame_info(i)
        img = load_greyscale(rec.path, normalise_01=self._normalise_01, dtype=self._dtype)
        return img, float(rec.timestamp), str(rec.frame_id)

    def iter(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ):
        if not isinstance(start, int) or start < 0:
            raise ValueError("start must be int >= 0")
        if stop is None:
            stopi = len(self._records)
        else:
            if not isinstance(stop, int) or stop < 0:
                raise ValueError("stop must be int >= 0 or None")
            stopi = min(int(stop), len(self._records))
        if not isinstance(step, int) or step <= 0:
            raise ValueError("step must be int > 0")

        i = int(start)
        while i < stopi:
            yield self.get(i)
            i += int(step)
