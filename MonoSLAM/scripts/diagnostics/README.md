# Diagnostics

Debugging tools for the current frontend. Unlike `scripts/demos/`, these expose
detailed options (PnP thresholds, gates, replay ranges) and are not minimal
usage examples. Run from the repository root. They need ETH3D or KITTI; see
`data/README.md`.

- `diag_pnp.py`: dataset-agnostic PnP diagnostic driver. It runs the keyframe
  pipeline over a frame range with configurable PnP gates, threshold-stability
  replay and per-frame scorecards. The two wrappers below supply a default
  profile; run directly, it requires `--profile`.
- `diag_pnp_eth3d.py`: `diag_pnp.py` with the ETH3D profile.
- `diag_pnp_kitti.py`: `diag_pnp.py` with the KITTI profile.

Run a wrapper with `--help` for the full option list. Output is written under
`out/`, which is git-ignored.
