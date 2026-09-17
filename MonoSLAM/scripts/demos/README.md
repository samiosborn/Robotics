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

## Website feature images

`export_feature_pair.py` reuses the dataset loader, frontend feature pipeline,
descriptor matcher and fundamental-matrix consensus without running the frontend.
From the repository root, with the ETH3D dataset installed:

```bash
uv run python scripts/demos/export_feature_pair.py --profile configs/profiles/eth3d_c2.yaml --i0 0 --i1 3 --max-draw 48 --draw-seed 0 --out-dir out/website_exports
```

This writes `eth3d_keypoints_pair.png`, `eth3d_matches_raw.png`,
`eth3d_matches_inliers.png` and `eth3d_feature_matching_results.json` into the
ignored output directory. No GUI or additional dependencies are required.
Frame indices address the complete timestamp-sorted sequence, as in the frontend
demo; profile start/step/max_frames do not resample this pair.

Frames 0–3 retain strong overlap with a more visible baseline than consecutive
frames: 213 tentative matches, 208 final consensus inliers and roughly 33 pixels
median tentative-match displacement with the current configuration. Their
timestamps are 11873.252219 and 11873.362809 seconds.

All descriptor-bearing multiscale Harris keypoints are shown, including detections
at different pyramid levels near the same location. Matching uses the frontend's
default BRIEF/Hamming settings. Verification uses the unchanged profile's 8-pixel
Sampson threshold, 2000 RANSAC trials and guarded refit. These are two-view feature
results, not a reconstructed map or independently established ground truth.

For legibility, the raw panel draws a seeded uniform sample of 48 matches; the
verified panel draws only the surviving inliers from that same sample. Both
panels disclose displayed and full counts. The JSON records inputs and hashes,
configuration, consensus statistics, display indices, command and source revision.
Change `--i0`/`--i1` to export another pair, or increase `--max-draw` to show more
correspondences; these options do not change matching or verification.
