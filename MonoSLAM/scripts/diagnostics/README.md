# Diagnostics

Reusable diagnostics for inspecting current frontend behaviour. Unlike
`scripts/demos/`, these are debugging tools with detailed CLI options
(thresholds, gates, replay ranges), not minimal usage examples. Run from
the repository root with `PYTHONPATH=.`.

- `diag_pnp.py` — dataset-agnostic PnP frontend diagnostic driver: runs the
  keyframe pipeline over a frame range with configurable PnP gates,
  threshold-stability replay, and per-frame scorecards.
- `diag_pnp_eth3d.py` — ETH3D-profile wrapper of `diag_pnp.py`.
- `diag_pnp_kitti.py` — KITTI-profile wrapper of `diag_pnp.py`.
- `diag_landmark_quality.py` — audits reprojection error and landmark
  quality over a frame range.
- `diag_seed_state.py` — dumps bootstrap/seed keyframe state for a frame
  range, for inspecting the map immediately after initialisation.
