# Agent brief

## Project
MonoSLAM is a monocular SLAM research codebase.
The current focus is frontend robustness: rescue-pose quality, downstream geometry integrity, and dataset-boundary hygiene ready for a second dataset.

## Trusted baseline
- BA-enabled pipeline with pose-eligible promotion guard, earlier rescued-support refresh, and a gated multi-seed 40 px canonical-pose proxy for bad rescue poses
- Stays healthy through frame 18; first failure at frame 19
- 22-frame stats (post canonical-pose-proxy patch): ETH3D ok=17/22, rescue=10/9, refresh=9; KITTI ok=16/22, rescue=11/5, refresh=2
- 81 tests pass

## Current first failure
- Frame 19: `pnp_ransac_failed` on live correspondences, rescue also fails
- Active keyframe: frame 18 (refreshed basis)
- Failure classification: coherent 2D tracks attached to a geometrically incompatible 3D support set

## Resolved: frame-16 / frame-12 canonical-pose quality
- The earlier leading interpretation (canonical rescue poses at frames 12 and 16 were sharp temporal outliers) was confirmed and fixed
- Production now runs a gated 40 px multi-seed re-solve on the rescue correspondence set whenever accepted-inlier residual median exceeds 8 px, and stores the best proxy pose in place of the raw rescue pose
- ETH3D frames 12 and 16 and KITTI frame 18 now trigger and select materially better canonical poses (see `exp/experiment_log.md`, "2026-06-20 - Canonical-pose proxy production patch")
- This did not change the frame-19 short-horizon failure — classified `result inconclusive` for that specific downstream survival question, but kept as a genuine canonical-pose-quality improvement

## Current open question
Frame 19 still fails after the canonical-pose-proxy patch. The bad-support-set question that caused frame 19 to fail is now a separate, open question from the (resolved) frame-16 pose-quality question.

## Best next step
Re-audit the frame-19 support set now that upstream canonical poses (12, 16, 18) are proxy-corrected — the original frame-19 diagnosis predates that fix and may no longer describe the current failure mode accurately.

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
The current frontier is understanding and correcting the frame-16 accepted rescue pose.

## Dataset-boundary state
- `src/slam/` is production, mostly dataset-agnostic
- `src/datasets/image_sequence.py` is the neutral sequence contract
- `src/datasets/eth3d.py` stays ETH3D-specific
- ETH3D entrypoints, defaults, visualisation, and experiment orchestration stay in scripts
- pure diagnostics may move from scripts into `src/slam/` only when they are reusable, pure, and stable
