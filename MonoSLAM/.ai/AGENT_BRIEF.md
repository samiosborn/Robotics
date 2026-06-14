# Agent brief

## Project
MonoSLAM is a monocular SLAM research codebase.
The current focus is frontend robustness: rescue-pose quality, downstream geometry integrity, and dataset-boundary hygiene ready for a second dataset.

## Trusted baseline
- BA-enabled pipeline with pose-eligible promotion guard and earlier rescued-support refresh
- Stays healthy through frame 18; first failure at frame 19
- 40-frame stats: ok=17/failed=23, promoted=2,4,6, rescues=10/9, BA=3/3
- 71 tests pass

## Current first failure
- Frame 19: `pnp_ransac_failed` on 22 live correspondences, rescue also fails
- Active keyframe: frame 18 (refreshed basis)
- All 22 correspondences are 2D-coherent (displacement consistency median 2.34 px)
- Every fixed-threshold replay (3/5/8/12 px) produces no inliers from 22 correspondences
- Failure classification: coherent 2D tracks attached to a geometrically incompatible 3D support set

## Leading interpretation
- Canonical rescue poses at frames 12 and 16 are sharp temporal outliers
- Frame 16 lies behind the frame-15-to-17 camera-centre chord; rotation-path excess 9.94 deg
- Frame 16 alone contributes 25 % of the 340-observation squared error and 31 % of residuals above 8 px
- Frames 12 and 16 together account for 73 % of residuals above 8 px in the live set
- A time-weighted frame-15-to-17 pose interpolation at frame 16 reduces its squared error by 71 %
- Rescue refresh is beneficial for support continuity; suppressing frame 16 refresh worsens pipeline survival
- Frame-16 accepted rescue pose is the main current outlier to understand

## Current open question
Why does the frame-16 20 px localisation-only rescue accept a pose far worse than the local frame-15-to-17 temporal interpolation on the same later-live landmarks?

## Best next step
Audit frame-16 rescue candidate generation and acceptance stages against the local temporal reference.

## Important files
Production:
- `src/slam/frame_pipeline.py`
- `src/slam/pnp_frontend.py`
- `src/slam/keyframe.py`
- `src/slam/bundle_adjustment.py`
- `src/slam/pnp_diagnostics.py`
- `src/slam/pnp_stats.py`

Datasets:
- `src/datasets/image_sequence.py`
- `src/datasets/eth3d.py`

Diagnostics and runners:
- `scripts/diag_pnp_eth3d.py`
- `scripts/diag_frame16_pose_quality.py`
- `scripts/demo_frontend_eth3d.py`
- `scripts/frontend_eth3d_common.py`
- `scripts/frontend_reporting.py`
- `scripts/jsonl_io.py`

Notes:
- `exp/current_status.md`
- `exp/experiment_log.md`

## Working mode
Default to:
- diagnosis first
- one narrow patch at a time
- validate every kept change
- update notes after meaningful runs

Do not assume the next fix is BA widening or rescue threshold relaxation.
The current frontier is understanding and correcting the frame-16 accepted rescue pose.

## Dataset-boundary state
- `src/slam/` is production, mostly dataset-agnostic
- `src/datasets/image_sequence.py` is the neutral sequence contract
- `src/datasets/eth3d.py` stays ETH3D-specific
- ETH3D entrypoints, defaults, visualisation, and experiment orchestration stay in scripts
- pure diagnostics may move from scripts into `src/slam/` only when they are reusable, pure, and stable
