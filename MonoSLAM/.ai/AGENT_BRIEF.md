# Agent brief

## Project
slam_from_scratch (Python distribution `MonoSLAM`) is a from-scratch monocular SLAM research codebase.
The current focus is frontend robustness: rescue-pose quality, downstream geometry integrity, and dataset-boundary hygiene.

## Trusted baseline
- BA-enabled pipeline with pose-eligible promotion guard, earlier rescued-support refresh, and a gated multi-seed 40 px canonical-pose proxy for bad rescue poses
- Local BA uses a hard monocular scale gauge (baseline of 2026-09-15)
- ETH3D `cables_2_mono`, attempted frames 2–41: 16 accepted, last accepted frame 17, first failure frame 18
- 92 tests pass
- Details: `exp/current_status.md`; history: `exp/experiment_log.md`

## Current first failure
- Frame 18: `pnp_ransac_failed`, 0 of 23 PnP inliers in the BA-on baseline
- Earlier frame-19 diagnoses in the notes predate the BA gauge fix and may not describe the current failure

## Resolved: canonical-pose quality at frames 12 and 16
- Canonical rescue poses at ETH3D frames 12 and 16 were temporal outliers, now corrected
- Production runs a gated 40 px multi-seed re-solve on the rescue correspondence set whenever the accepted-inlier residual median exceeds 8 px, and stores the best proxy pose in place of the raw rescue pose
- See `exp/experiment_log.md`, "2026-06-20 - Canonical-pose proxy production patch"

## Open questions
- Cause of the first failure under the BA-on baseline; the earlier frame-19 classification was coherent 2D tracks attached to a geometrically incompatible 3D support set
- Weak-window BA conditioning: the `[1,2,3]` BA event is conditioning-sensitive. It is the next causal measurement target if broader validation exposes regressions; no BA admission gate is justified yet

## Best next step
Re-audit the first-failure support set under the current BA-on baseline before proposing any production change.

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
- `scripts/diagnostics/diag_pnp.py`
- `scripts/diagnostics/diag_pnp_eth3d.py`
- `scripts/diagnostics/diag_pnp_kitti.py`
- `scripts/demos/demo_frontend_eth3d.py`
- `scripts/frontend_common.py`
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

## Dataset-boundary state
- `src/slam/` is production, mostly dataset-agnostic
- `src/datasets/image_sequence.py` is the neutral sequence contract
- `src/datasets/eth3d.py` stays ETH3D-specific
- ETH3D entrypoints, defaults, visualisation, and experiment orchestration stay in scripts
- pure diagnostics may move from scripts into `src/slam/` only when they are reusable, pure, and stable
