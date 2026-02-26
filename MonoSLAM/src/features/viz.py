# src/features/viz.py

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw


# Draw matches on a side-by-side image and save.
def draw_matches(
    imgA: Image.Image,
    imgB: Image.Image,
    kpsA: np.ndarray,
    kpsB: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    out_path: Path,
    *,
    max_draw: int = 200,
    draw_topk: int | None = None,
    draw_inliers_only: bool = False,
    inlier_mask: np.ndarray | None = None,
    r: int = 3,
) -> None:
    out_path = Path(out_path)
    A = imgA.convert("RGB")
    B = imgB.convert("RGB")
    WA, HA = A.size
    WB, HB = B.size
    canvas = Image.new("RGB", (WA + WB, max(HA, HB)))
    canvas.paste(A, (0, 0))
    canvas.paste(B, (WA, 0))
    draw = ImageDraw.Draw(canvas)

    kpsA = np.asarray(kpsA)
    kpsB = np.asarray(kpsB)
    ia = np.asarray(ia)
    ib = np.asarray(ib)
    if ia.ndim != 1 or ib.ndim != 1:
        raise ValueError("ia and ib must be 1D index arrays")
    if ia.shape[0] != ib.shape[0]:
        raise ValueError("ia and ib must have the same length")

    sel = np.arange(int(ia.size), dtype=np.int64)

    if bool(draw_inliers_only):
        if inlier_mask is None:
            sel = np.zeros((0,), dtype=np.int64)
        else:
            m = np.asarray(inlier_mask, dtype=bool)
            if m.ndim != 1 or m.shape[0] != ia.shape[0]:
                raise ValueError("inlier_mask must be shape (M,) matching ia/ib")
            sel = sel[m]

    if draw_topk is not None:
        k = int(draw_topk)
        if k < 0:
            raise ValueError(f"draw_topk must be >= 0 or None; got {draw_topk}")
        sel = sel[:k]

    md = int(max_draw)
    if md < 0:
        raise ValueError(f"max_draw must be >= 0; got {max_draw}")
    sel = sel[:md]

    rr = int(r)
    for idx in sel:
        i = int(ia[idx])
        j = int(ib[idx])
        x0 = float(kpsA[i, 0])
        y0 = float(kpsA[i, 1])
        x1 = float(kpsB[j, 0]) + float(WA)
        y1 = float(kpsB[j, 1])
        draw.ellipse((x0 - rr, y0 - rr, x0 + rr, y0 + rr))
        draw.ellipse((x1 - rr, y1 - rr, x1 + rr, y1 + rr))
        draw.line((x0, y0, x1, y1), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


# Draw projected box corners from img0 onto img1 using homography H.
def draw_projected_box(
    img0: Image.Image,
    img1: Image.Image,
    H: np.ndarray,
    out_path: Path,
    *,
    width: int = 3,
) -> None:
    out_path = Path(out_path)
    scene = img1.convert("RGB")
    H = np.asarray(H, dtype=float)
    W0, H0 = img0.size

    x = np.array(
        [
            [0.0, float(W0 - 1), float(W0 - 1), 0.0],
            [0.0, 0.0, float(H0 - 1), float(H0 - 1)],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    y = H @ x
    w = y[2, :]
    w = np.where(np.abs(w) > 1e-12, w, np.nan)
    y = y[:2, :] / w
    quad = [(float(y[0, i]), float(y[1, i])) for i in range(y.shape[1])]

    draw = ImageDraw.Draw(scene)
    draw.polygon(quad, outline=(255, 0, 0), width=int(width))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.save(str(out_path))
