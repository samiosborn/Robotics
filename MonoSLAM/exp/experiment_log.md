## 2026-06-12 — low-cardinality rescue retry

Base commit: 5a1ad85

Hypothesis
Frame 19 fails because low-cardinality 40 px rescue search is seed-fragile.

Change
Added bounded multi-seed retry in second-stage seeded 40 px rescue.

Validation
- uv run python -m pytest tests/slam -q
- PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 20 ...
- PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py

Result
No improvement. Frame 19 still failed with pnp_ransac_failed and 0/22 PnP inliers.

Decision
Reverted.

Next
Diagnose why coherent loose 40 px support is not turning into usable rescue.