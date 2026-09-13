# MonoSLAM

A from-scratch monocular SLAM implementation in Python/NumPy, built as a
research and learning project. Feature detection, two-view geometry, PnP
localisation, and a keyframe-based frontend are implemented and tested from
first principles rather than wrapping an existing SLAM library.

## Status

This is a research/learning implementation, not a production SLAM system.

- The keyframe frontend, with rescue-based re-localisation, canonical-pose
  correction, and local bundle adjustment, runs stably for the first ~18
  frames of the bundled ETH3D test sequence before hitting a tracking
  failure that is still under investigation.
- Long-horizon mapping and full bundle adjustment (beyond the current
  per-keyframe local BA) are not yet robust.
- See `exp/current_status.md` for the current trusted baseline and
  `exp/experiment_log.md` for the full experiment history.

## Capabilities

- **Feature detection and matching** — Harris keypoints, multiscale
  pyramids, BRIEF descriptors, NCC and Hamming-distance matching
  (`src/features/`).
- **Two-view geometry** — normalised 8-point fundamental/essential matrix
  estimation, RANSAC, pose recovery via cheirality (`src/geometry/`).
- **Triangulation** — 3D point recovery from two calibrated views
  (`src/geometry/triangulation.py`).
- **Keyframe-based monocular frontend** — bootstrap, tracking, rescue
  re-localisation, and keyframe promotion (`src/slam/`).
- **PnP localisation** — RANSAC PnP with spatial and displacement-
  consistency gating (`src/geometry/pnp.py`, `src/slam/pnp_frontend.py`).
- **Local bundle adjustment** — minimal BA on promoted keyframes
  (`src/slam/bundle_adjustment.py`).
- **Dataset adapters** — ETH3D SLAM and KITTI odometry sequence loaders
  (`src/datasets/`).

## Repository structure

```
configs/        camera intrinsics and dataset/run profiles
data/           small checked-in example data (see data/README.md)
notebooks/      worked-example notebooks and LaTeX derivation notes
scripts/
  demos/        runnable examples of MonoSLAM functionality
  diagnostics/  reusable debugging tools for the current frontend
src/            production implementation
tests/          pytest suite
exp/            development log (trusted baseline + experiment history)
```

## Installation

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Two-view demo

Runs immediately, no dataset download required:

```bash
PYTHONPATH=. uv run python scripts/demos/demo_match_two_frames.py data/two-view/box.png data/two-view/box_in_scene.png
```

Also try the synthetic two-view geometry demos, which need no images at all:

```bash
PYTHONPATH=. uv run python scripts/demos/demo_synthetic_two_view.py
PYTHONPATH=. uv run python scripts/demos/demo_fundamental_8point.py
```

## Dataset setup (ETH3D / KITTI)

The frontend demos need a real image sequence. See
[`data/README.md`](data/README.md) for expected directory layouts and
download instructions for ETH3D and KITTI. Datasets are not committed to
this repository.

## Frontend demo commands

```bash
PYTHONPATH=. uv run python scripts/demos/demo_frontend_eth3d.py
PYTHONPATH=. uv run python scripts/demos/demo_frontend_kitti.py
```

For debugging the frontend's PnP/rescue behaviour in more detail, see
`scripts/diagnostics/` (e.g. `diag_pnp_eth3d.py`, `diag_pnp_kitti.py`).

## Tests

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests -q
```
