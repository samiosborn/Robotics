# slam_from_scratch

A from-scratch monocular SLAM implementation in Python and NumPy, built as a
research and learning project. Feature detection, two-view geometry, PnP
localisation, a keyframe-based frontend and local bundle adjustment are
implemented directly rather than wrapping an existing SLAM library.

The Python distribution and import packages are named `MonoSLAM`.

## Status

This is a research implementation, not a production SLAM system.

- On the ETH3D `cables_2_mono` sequence the frontend, with local bundle
  adjustment enabled, accepts frames 2 to 17 and first fails at frame 18
  (PnP RANSAC failure). The cause is still under investigation.
- Mapping over longer horizons is not robust, and there is no global bundle
  adjustment or loop closure.
- `exp/current_status.md` holds the trusted baseline and
  `exp/experiment_log.md` the full experiment history.

## Capabilities

- **Feature detection and matching**: multiscale Harris keypoints, BRIEF
  descriptors, NCC and Hamming-distance matching (`src/features/`).
- **Two-view geometry**: normalised 8-point fundamental and essential matrix
  estimation, RANSAC, homography, and pose recovery by cheirality
  (`src/geometry/`).
- **Triangulation**: 3D points from two calibrated views
  (`src/geometry/triangulation.py`).
- **PnP localisation**: RANSAC PnP with spatial and displacement-consistency
  gating (`src/geometry/pnp.py`, `src/slam/pnp_frontend.py`).
- **Keyframe frontend**: bootstrap, tracking, rescue re-localisation and
  keyframe promotion (`src/slam/`).
- **Local bundle adjustment**: run on promoted keyframes, with a fixed
  monocular scale gauge (`src/slam/bundle_adjustment.py`).
- **Dataset adapters**: ETH3D SLAM and KITTI odometry sequences
  (`src/datasets/`).

## Repository layout

```text
configs/        camera intrinsics, feature/bootstrap settings, dataset profiles
data/           tracked two-view example images; external datasets go here
exp/            trusted baseline and experiment log
notebooks/      example notebooks and LaTeX derivation notes
scripts/
  demos/        runnable examples
  diagnostics/  frontend PnP/rescue diagnostics
src/            implementation
tests/          pytest suite
```

## Setup

Requires Python 3.10 or later and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run everything from the repository root. Generated output is written to `out/`,
which is git-ignored.

## Demos without external data

```bash
uv run python scripts/demos/demo_match_two_frames.py data/two_view/box.png data/two_view/box_in_scene.png
uv run python scripts/demos/demo_synthetic_two_view.py
uv run python scripts/demos/demo_fundamental_8point.py
uv run python scripts/demos/demo_fundamental_ransac_8point.py
```

`demo_match_two_frames.py` defaults to NCC matching; add `--match brief` for
BRIEF descriptors.

## Frontend demos and diagnostics

These need ETH3D or KITTI, which are not distributed with the repository. See
[`data/README.md`](data/README.md) for the expected layout.

```bash
uv run python scripts/demos/demo_frontend_eth3d.py
uv run python scripts/demos/demo_frontend_kitti.py
uv run python scripts/diagnostics/diag_pnp_eth3d.py
```

`--num_track` sets how many frames are processed after bootstrap (default 5).
Both dataset profiles also set `max_frames: 10`, which caps the run; to go
further, pass a copy of the profile with a larger value using `--profile`.

See [`scripts/demos/README.md`](scripts/demos/README.md) and
[`scripts/diagnostics/README.md`](scripts/diagnostics/README.md) for details.

## Tests

```bash
uv run pytest tests -q
```

The suite uses no external data.
