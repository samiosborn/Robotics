# Demos

Runnable examples of the implementation. Run from the repository root; see the
top-level README for setup.

Without external data:

- `demo_synthetic_two_view.py`: essential-matrix estimation and pose
  decomposition on synthetic correspondences.
- `demo_fundamental_8point.py`: 8-point fundamental matrix estimation on
  synthetic correspondences.
- `demo_fundamental_ransac_8point.py`: RANSAC variant of the above.
- `demo_match_two_frames.py`: keypoint detection and matching between two
  images, with a homography fit. Takes two image paths, for example
  `data/two_view/box.png data/two_view/box_in_scene.png`. `--match brief`
  selects BRIEF descriptors instead of NCC.

With ETH3D or KITTI (see `data/README.md`):

- `demo_frontend.py`: dataset-agnostic keyframe frontend demo. The two wrappers
  below supply a default profile; run directly, it requires `--profile`.
- `demo_frontend_eth3d.py`: frontend on the ETH3D `cables_2_mono` sequence.
- `demo_frontend_kitti.py`: frontend on KITTI odometry sequence 00.
- `export_feature_pair.py`: feature matching figures for one frame pair
  (described below).

## Feature pair export

`export_feature_pair.py` runs the frontend's feature pipeline, descriptor
matcher and fundamental-matrix consensus on two frames of a profile's sequence,
without running the frontend itself.

```bash
uv run python scripts/demos/export_feature_pair.py \
  --profile configs/profiles/eth3d_c2.yaml --i0 0 --i1 3 \
  --max-draw 48 --draw-seed 0 --out-dir out/website_exports
```

It writes `eth3d_keypoints_pair.png`, `eth3d_matches_raw.png`,
`eth3d_matches_inliers.png` and `eth3d_feature_matching_results.json`.

- Frame indices address the full timestamp-sorted sequence; the profile's
  `start`, `step` and `max_frames` do not apply.
- All descriptor-bearing multiscale Harris keypoints are drawn, including
  detections of the same location at different pyramid levels.
- Matching uses the frontend's default BRIEF/Hamming settings. Verification
  uses the profile's 8 px Sampson threshold, 2000 RANSAC trials and a guarded
  refit.
- The raw panel shows a seeded uniform sample of `--max-draw` matches. The
  verified panel shows the inliers within that sample. Both state the displayed
  and total counts.
- The JSON records input hashes, configuration, consensus statistics, the
  displayed indices, the command and the source revision.
- `--i0`, `--i1`, `--max-draw` and `--draw-seed` change only the pair and the
  drawing, not matching or verification.

For frames 0 and 3 of `cables_2_mono` with the current configuration, there are
213 tentative matches and 208 consensus inliers. These are two-view feature
statistics, not a reconstructed map and not independently established ground
truth.
