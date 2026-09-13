# Demos

Runnable, reusable demonstrations of MonoSLAM functionality. Run from the
repository root with `PYTHONPATH=.` (see the top-level README).

- `demo_synthetic_two_view.py` — synthetic two-view essential-matrix
  estimation and pose decomposition, no dataset required.
- `demo_fundamental_8point.py` — synthetic two-view 8-point fundamental
  matrix estimation.
- `demo_fundamental_ransac_8point.py` — RANSAC variant of the 8-point demo.
- `demo_match_two_frames.py` — feature detection and matching between two
  arbitrary images (defaults work with `data/two-view/`).
- `demo_frontend.py` — dataset-agnostic keyframe frontend demo engine, used
  by the two dataset-specific wrappers below.
- `demo_frontend_eth3d.py` — frontend demo on the ETH3D `cables_2_mono`
  sequence (see `data/README.md` for dataset setup).
- `demo_frontend_kitti.py` — frontend demo on a KITTI odometry sequence
  (see `data/README.md` for dataset setup).
