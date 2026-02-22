# src/features/viz.py

from __future__ import annotations

from pathlib import Path

import numpy as np

from PIL import Image
from PIL import ImageDraw


# Draw matches on a side-by-side image and save
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
    r: int = 3,
):

    # Convert output path
    out_path = Path(out_path)

    # Ensure RGB for drawing
    A = imgA.convert("RGB")
    B = imgB.convert("RGB")

    # Read sizes
    WA, HA = A.size
    WB, HB = B.size

    # Create a canvas side-by-side
    canvas = Image.new("RGB", (WA + WB, max(HA, HB)))

    # Paste images
    canvas.paste(A, (0, 0))
    canvas.paste(B, (WA, 0))

    # Prepare drawing context
    draw = ImageDraw.Draw(canvas)

    # Make sure arrays are numpy
    kpsA = np.asarray(kpsA)
    kpsB = np.asarray(kpsB)
    ia = np.asarray(ia)
    ib = np.asarray(ib)

    # Limit number of matches drawn
    M = int(min(int(ia.size), int(max_draw)))

    # Draw each match
    for m in range(M):

        # Read match indices
        i = int(ia[m])
        j = int(ib[m])

        # Read keypoint coordinates (x,y) in image A
        x0 = float(kpsA[i, 0])
        y0 = float(kpsA[i, 1])

        # Read keypoint coordinates (x,y) in image B (offset by WA on canvas)
        x1 = float(kpsB[j, 0]) + float(WA)
        y1 = float(kpsB[j, 1])

        # Draw a small circle at keypoint A
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r))

        # Draw a small circle at keypoint B
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r))

        # Draw a line between matched keypoints
        draw.line((x0, y0, x1, y1), width=1)

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    canvas.save(str(out_path))


# Draw projected box corners from img0 onto img1 using homography H
def draw_projected_box(
    img0: Image.Image,
    img1: Image.Image,
    H: np.ndarray,
    out_path: Path,
    *,
    width: int = 3,
):

    # Convert output path
    out_path = Path(out_path)

    # Ensure RGB for drawing
    scene = img1.convert("RGB")

    # Convert H to numpy
    H = np.asarray(H, dtype=float)

    # Get source image size (box image)
    W0, H0 = img0.size

    # Define source corners in image0 coordinates -> shape (3,4) homogeneous
    x = np.array(
        [
            [0.0, float(W0 - 1), float(W0 - 1), 0.0],
            [0.0, 0.0, float(H0 - 1), float(H0 - 1)],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    # Project corners into scene using homography
    y = H @ x

    # Dehomogenise (guard divide-by-zero)
    w = y[2, :]
    w = np.where(np.abs(w) > 1e-12, w, np.nan)
    y = y[:2, :] / w

    # Convert to list of tuples for PIL polygon
    quad = [(float(y[0, i]), float(y[1, i])) for i in range(y.shape[1])]

    # Prepare drawing context
    draw = ImageDraw.Draw(scene)

    # Draw polygon outline (red)
    draw.polygon(quad, outline=(255, 0, 0), width=int(width))

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save scene overlay
    scene.save(str(out_path))
