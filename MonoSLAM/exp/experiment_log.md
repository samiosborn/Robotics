# MonoSLAM experiment log

## 2026-06-12 — Seed mutation cleanup for support-basis helpers

Base state
- pre-BA structural cleanup phase

Hypothesis
- support-basis helper paths may be mutating nested seed state through shallow copies

Change
- updated support-basis helper copying to avoid shared nested mutation
- added regression coverage in `tests/slam/test_keyframe_state.py`

Validation
- `uv run python -m pytest tests/slam/test_keyframe_state.py -q`
- `uv run python -m pytest tests/slam -q`

Result
- tests passed
- mutation hazard removed

Decision
- kept

---

## 2026-06-12 — Canonical pose agreement in keyframes

Base state
- pre-BA API/invariant hardening

Hypothesis
- future BA could update canonical pose storage while stale mirrored keyframe pose state remains silently usable

Change
- added pose mirror consistency guard in `src/slam/keyframe_state.py`
- added regression test for stale `seed["poses"][kf]` vs `record["pose"]`

Validation
- `uv run python -m pytest tests/slam/test_keyframe_state.py -q`
- `uv run python -m pytest tests/slam -q`
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py`

Result
- runtime behaviour unchanged for valid seeds
- inconsistent mirrored pose state now fails loudly

Decision
- kept

---

## 2026-06-12 — First minimal local BA

Base state
- frontend baseline after pre-BA cleanup

Hypothesis
- a small local BA pass can improve local geometry safely without destabilising the frontend

Change
- added `src/slam/bundle_adjustment.py`
- inserted local BA on newly promoted non-rescue keyframes
- optimised small recent keyframe window with LM
- fixed oldest local keyframe as anchor
- added synthetic BA unit test

Validation
- `uv run python -m pytest tests/slam -q`
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 20 ...`

Result
- BA attempted / succeeded: 4 / 4 on short validation
- local mean and median reprojection error improved on every attempted BA pass
- broader short scorecard stayed roughly neutral
- invariants and pose ownership stayed intact

Decision
- kept

---

## 2026-06-12 — BA propagation diagnosis

Base state
- minimal local BA enabled

Hypothesis
- BA may be improving local geometry but not the support consumed by later frames

Change
- diagnostic-only extension to `scripts/diag_pnp_eth3d.py` for BA landmark propagation

Validation
- BA-enabled ETH3D scorecard and JSONL parsing

Result
- BA-improved landmarks were reused later
- later failure support was mostly inside the latest BA-refined landmark set
- BA improved local geometry but weakly propagated to later survival

Decision
- no production change
- use result to guide next diagnosis

---

## 2026-06-12 — Pose-eligible promotion guard

Base state
- post-kf7 collapse diagnosed as lookup / basis starvation

Hypothesis
- current promotion logic allows a frame to become active keyframe with enough linked landmarks, but not enough pose-eligible linked landmarks

Change
- promotion now requires enough pose-eligible linked landmarks
- pose-eligible means linked, valid finite `X_w`, and enough observations under the same origin-aware support rule as PnP

Validation
- `uv run python -m pytest tests/slam -q`
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 14 ...`

Result
- frame 7 no longer promoted
- frame 8 improved from lookup starvation to strong support:
  - trusted old collapse: 430 raw -> 108 mapped -> 32 obs-pass
  - new run: 276 raw -> 208 mapped -> 91 obs-pass, 90/91 PnP
- earlier healthy frames stayed aligned

Decision
- kept

---

## 2026-06-12 — Earlier rescued-support refresh

Base state
- after promotion guard, new downstream bottleneck appeared at frame 12
- rescue succeeded through frames 8–11 but active basis was not refreshed before failure

Hypothesis
- the hard `current_kf < 14` guard delays rescued-support refresh too long

Change
- removed only the hard `current_kf < 14` delay from guarded rescued-support refresh
- kept the existing support-strength checks intact

Validation
- `uv run python -m pytest tests/slam -q`
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 14 ...`

Result
- rescued frames now refresh when support is strong enough
- frame 12 no longer collapses
- 14-frame diagnostic became fully clean: `ok=14 failed=0`

Decision
- kept

---

## 2026-06-12 — Longer BA-enabled scorecard after early refresh

Base state
- promotion guard plus earlier rescued-support refresh enabled

Hypothesis
- early rescued-support refresh may materially delay the first collapse

Change
- diagnosis only

Validation
- `uv run python scripts/diag_pnp_eth3d.py --num_track 40 ...`
- focused replay to first failure

Result
- first collapse moved from frame 12 to frame 19
- long-run summary:
  - ok / failed: 17 / 23
  - rescue attempted / succeeded: 10 / 9
  - support refresh triggered through frame 18
- frame 19:
  - active keyframe 18
  - 616 tracked pairs
  - 22 mapped / valid / observation-gated / final PnP correspondences
  - 0 PnP inliers
  - rescue attempted and failed

Decision
- no production change
- next focus moved to frame-19 low-cardinality failure

---

## 2026-06-12 — Low-cardinality rescue retry at frame 19

Base state
- frame 19 classified as likely low-cardinality PnP rescue problem

Hypothesis
- second-stage seeded 40 px rescue is seed-fragile on the 22-point frame-19 support set

Change
- bounded multi-seed retry added to low-cardinality seeded 40 px rescue path in `src/slam/pnp_frontend.py`

Validation
- `uv run python -m pytest tests/slam -q`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 20 ...`
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py`

Result
- no improvement
- frame 19 still failed with `pnp_ransac_failed` and `0/22`
- earlier behaviour stayed unchanged

Decision
- reverted / not kept

---

## 2026-06-12 — Live-pipeline frame-19 diagnostics

Base state
- existing threshold-pair diagnostics could inspect stale promoted-keyframe bundles instead of the live refreshed active basis

Hypothesis
- the frame-19 interpretation may be distorted unless diagnostics use the actual live pipeline bundle

Change
- diagnostic-only extension to `scripts/diag_pnp_eth3d.py`
- added live-pipeline bundle replay using `frontend_out["pose_out"]["corrs"]`
- added live threshold and threshold-pair events

Validation
- `uv run python -m py_compile scripts/diag_pnp_eth3d.py`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 19 --scorecard long --threshold_pair_frame_index 19 ...`
- JSONL parsing

Result
- live diagnostics now match the actual frame-19 refreshed frame-18 active basis
- frame 19 live summary:
  - 22 live correspondences
  - pipeline `pnp_ransac_failed`, `0/22`
  - live 3/5/8/12 px replay still fails
  - live 8/12 threshold-pair replay rejects all 22 correspondences

Decision
- diagnostic improvement kept
- notes updated to rely on live pipeline bundle, not stale diagnostic bundle

---

## 2026-06-13 — Live frame-19 displacement consistency

Base state
- trusted BA-enabled baseline with refreshed frame-18 active basis
- frame 19 fails pipeline and fixed 3/5/8/12 px PnP on 22 live correspondences

Hypothesis
- the 22 spatially coherent links may still contain an inconsistent frame-18-to-19 displacement field

Diagnostic step
- added live-pipeline local displacement-consistency reporting to `scripts/diag_pnp_eth3d.py`
- replayed frame 19 without applying the filter to pipeline state

Validation
- `uv run python -m py_compile scripts/diag_pnp_eth3d.py`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 19 --scorecard long --threshold_pair_frame_index 19 --out_dir /tmp/diag_pnp_eth3d_frame19_live_consistency`
- parsed live threshold and local-consistency events from `pnp_diag.jsonl`

Result
- all 22 live correspondences passed local displacement consistency
- median residual was 2.34 px and p90 residual was 4.72 px
- each fixed threshold produced 902 valid models from 1000 trials
- best support remained one point at 3/5/8/12 px
- failure is not low-cardinality search failure or incoherent 2D tracking
- the live 2D tracks are attached to a geometrically incompatible 3D support set

Decision
- diagnosis only
- kept the narrow live diagnostic
- production baseline unchanged

Next step
- compare the 22 links against the accepted frame-18 pose and stored frame-18 observations
- distinguish feature-to-landmark assignment error from landmark-geometry drift

---

## 2026-06-13 — Rescue-refresh suppression counterfactual

Base state
- trusted BA-enabled pipeline with first failure at frame 19
- bad canonical rescue poses at frames 12 and 16 were the main geometry-history outliers

Hypothesis
- frames 12 and 16 mainly damage downstream support when their rescued poses become refreshed active bases

Diagnostic step
- extended `scripts/diag_pnp_eth3d.py` with selected-frame refresh suppression
- kept rescue localisation, observation append, canonical pose storage, and rescue bookkeeping unchanged
- replayed 40 frames with refresh suppressed at frames 12 and 16

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 40 --scorecard off --threshold_pair_frame_index 19 --out_dir /tmp/diag_refresh_counterfactual_baseline`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 40 --scorecard off --threshold_pair_frame_index 19 --suppress_support_refresh_frames 12 16 --out_dir /tmp/diag_refresh_counterfactual_no12_no16`

Result
- frame 12 rescue stayed accepted at 40 / 45, with its refresh suppressed
- frame 13 and frame 14 refreshed normally
- frame 15 rescue weakened to 18 / 34 and did not refresh
- first failure moved earlier from frame 19 to frame 16 at 0 / 22
- frame-16 refresh suppression was not reached because rescue failed
- 40-frame ok / failed changed from 17 / 23 to 14 / 26
- frame-19 canonical-history p90 remained broad at 11.82 px with 14 / 18 landmarks drifting
- frame 12 still contributed 41.20 per cent of squared error and 56.00 per cent of residuals above 8 px

Decision
- bad rescue poses are correlated but not the main propagation path
- frame-12 active-basis refresh is beneficial for support continuity
- no production change

Next step
- isolate frame 16 with a frame-16-only no-refresh replay

---

## 2026-06-13 — Frame-16-only refresh suppression

Base state
- trusted BA-enabled pipeline with first failure at frame 19
- frame-12 refresh known to be useful for support continuity

Hypothesis
- frame-16 rescued-support refresh independently propagates downstream harm

Diagnostic step
- used the existing selected-frame refresh suppression replay
- preserved frame-12 refresh
- suppressed active-basis refresh only at frame 16
- kept rescue acceptance, observation append, canonical pose storage, and bookkeeping unchanged

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 40 --scorecard off --threshold_pair_frame_index 19 --out_dir /tmp/diag_refresh_frame16_baseline`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 40 --scorecard off --threshold_pair_frame_index 19 --suppress_support_refresh_frames 16 --out_dir /tmp/diag_refresh_frame16_no16`
- parsed frame summaries and frame-19 live assignment audits from both JSONL logs

Result
- baseline reproduced first failure at frame 19 and 17 / 40 frames ok
- frame 16 remained accepted at 24 / 28, with basis 15 retained
- first failure moved earlier to frame 17 at 0 / 16
- 40-frame ok / failed changed from 17 / 23 to 15 / 25
- counterfactual frame 19 still failed at 0 / 22 from active basis 15
- pooled canonical-history median / p90 changed from 2.87 / 10.87 px to 2.27 / 10.71 px
- drifting landmarks changed from 22 / 22 to 21 / 22
- frame 16 still contributed 24.62 per cent of squared error and 35.00 per cent of residuals above 8 px
- all 16 shared frame-19 landmarks remained drifting, with p90 changing from 11.00 px to 11.04 px

Decision
- frame-16 refresh is not the main problem
- suppressing it materially worsens support survival
- keep rescue refresh enabled
- no production change

Next step
- diagnose frame-16 rescue-pose acceptance quality against neighbouring canonical poses

---

## 2026-06-14 — Frame-16 refresh suppression reproduction

Base state
- trusted BA-enabled pipeline with first failure at frame 19
- frame-12 rescued-support refresh preserved

Diagnostic step
- reran the trusted 40-frame baseline
- suppressed active-basis refresh only at frame 16
- kept rescue acceptance, observation append, canonical pose storage, and bookkeeping unchanged

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 40 --scorecard off --threshold_pair_frame_index 19 --out_dir /tmp/diag_refresh_frame16_20260614_baseline`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 40 --scorecard off --threshold_pair_frame_index 19 --suppress_support_refresh_frames 16 --out_dir /tmp/diag_refresh_frame16_20260614_no16`
- parsed frame summaries and frame-19 live assignment audits from both JSONL logs

Result
- baseline reproduced first failure at frame 19 and 17 / 40 frames ok
- frame 16 remained accepted at 24 / 28, with basis 15 retained
- first failure moved earlier to frame 17 at 0 / 16
- pooled frame-19 canonical-history median / p90 changed from 2.87 / 10.87 px to 2.27 / 10.71 px
- 21 / 22 counterfactual frame-19 live landmarks remained drifting
- frame 16 contributed 24.62 per cent of squared error from 6.27 per cent of observations
- all 16 shared frame-19 landmarks remained drifting, with p90 changing from 11.00 px to 11.04 px

Decision
- frame-16 refresh is not the main problem
- suppressing it worsens support survival
- keep rescue refresh enabled
- no production change

---

## 2026-06-14 — Frame-16 accepted rescue-pose quality

Base state
- trusted BA-enabled pipeline with first failure at frame 19
- frame-16 refresh known to support downstream continuity

Diagnostic step
- added `scripts/diag_frame16_pose_quality.py`
- replayed through frame 19 without changing production state
- matched the exact 22 frame-19 live landmarks from the trusted assignment audit
- compared canonical frames 15–18 and a time-weighted frame-15-to-17 pose interpolation

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python -m py_compile scripts/diag_frame16_pose_quality.py`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_frame16_pose_quality.py --out /tmp/frame16_pose_quality.json`
- exact live-landmark id comparison against the baseline frame-19 JSONL audit

Result
- frame 16 was accepted as a 20 px localisation-only rescue with 24 / 28 inliers
- frame 16 lies behind the frame-15-to-17 camera-centre chord
- local camera-motion turn was 135.68 deg and rotation-path excess was 9.94 deg
- frame-16 median / p90 residual was 10.97 / 17.51 px on the 22 live landmarks
- frame 16 contributed 56.34 per cent of frames 15–18 squared error
- time interpolation reduced frame-16 median / p90 to 5.95 / 8.16 px
- interpolation reduced frame-16 squared error by 71.30 per cent
- replacing only frame 16 reduced full-history p90 from 10.87 px to 8.48 px
- full-history squared error fell by 18.13 per cent

Decision
- frame-16 accepted rescue pose is a main outlier
- it materially amplifies later geometry-history error but is not the sole cause
- no production change

Next step
- audit frame-16 rescue candidate generation and acceptance against the local temporal reference

---

## 2026-06-14 — Stage 1 dataset-boundary cleanup

Change
- moved generic frontend reporting and JSONL helpers out of `frontend_eth3d_common.py`
- replaced the duplicated diagnostic landmark index with `build_landmark_id_index`
- kept ETH3D loading and runner behaviour unchanged

Validation
- script compilation passed
- `tests/slam`: 66 passed
- frontend demo completed normally
- 12-frame PnP diagnostic: 12 ok, 0 failed

Decision
- kept

---

## 2026-06-14 — Stage 2 diagnostics-boundary cleanup

Change
- moved pure PnP reprojection, threshold-mask, and spatial-summary calculations into `src/slam/pnp_diagnostics.py`
- kept ETH3D orchestration, visualisation, replay, and experiment-specific diagnostics in `scripts/diag_pnp_eth3d.py`

Validation
- script compilation passed
- `tests/slam`: 71 passed
- frontend demo completed normally
- 12-frame PnP diagnostic: 12 ok, 0 failed

Decision
- kept

---

## 2026-06-14 — Stage 3 runner-level dataset-loader cleanup

Change
- added `src/datasets/loader.py` with explicit dispatch on `dataset.name`
- `"eth3d"` routes to existing `load_eth3d_sequence`; anything else raises `ValueError`
- updated `scripts/demo_frontend_eth3d.py` to import `load_sequence` and read `dataset_name` from `dataset_cfg["name"]`
- updated `scripts/diag_pnp_eth3d.py` identically
- removed hard-coded "ETH3D" from two error messages in the runners
- `eth3d.py`, `frontend_eth3d_common.py`, SLAM logic, and old experiment scripts left untouched

Validation
- script compilation passed: `src/datasets/*.py`, both runners, `frontend_eth3d_common.py`
- `tests/slam`: 71 passed
- frontend demo completed normally
- 12-frame PnP diagnostic: 12 ok, 0 failed

Decision
- kept

---

## 2026-06-14 — Minimum KITTI odometry adapter

Change
- added `src/datasets/kitti.py` with `load_kitti_sequence`
  - loads grayscale left images from `sequences/<seq>/image_0/*.png`
  - sorts by numeric stem; uses frame index as timestamp
  - returns `ImageSequence` with name `kitti:<seq>`
- added `kitti` dispatch branch to `src/datasets/loader.py`
- added `configs/cameras/kitti_odometry.yaml` with sequence-00 P0 intrinsics
- added `configs/profiles/kitti_odometry_00.yaml` referencing the new camera config
- added `tests/datasets/test_kitti.py` with 10 tests covering ordering, timestamps, frame ids, dispatch, and error paths
- no pose parsing, stereo, distortion, or Velodyne support added

Validation
- `uv run python -m py_compile src/datasets/*.py scripts/demo_frontend_eth3d.py scripts/diag_pnp_eth3d.py` passed
- `uv run python -m pytest tests/slam tests/datasets -q`: 81 passed, 0 failed
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py` completed normally
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 12 --scorecard short --threshold_pair_frame_index 9999 --out_dir /tmp/diag_pnp_eth3d_kitti_port_regression`: 12 ok, 0 failed

Decision
- kept

---

## 2026-06-14 — Real KITTI sequence 00 smoke test

Base state
- committed minimum KITTI odometry adapter

Real-data check
- selectively extracted all 4,541 sequence-00 `image_0` PNGs from the official KITTI grayscale odometry archive
- confirmed the archive layout `dataset/sequences/00/image_0/*.png`
- confirmed the adapter loaded IDs `000000` to `004540` as 1241 by 376 greyscale frames

Smoke test
- default bootstrap `0 -> 1` failed the existing parallax gate at 0.60 degrees median
- bootstrap `0 -> 2` also failed at 0.90 degrees median
- bootstrap `0 -> 3` passed at 1.46 degrees median with 427 initial landmarks
- frame 4 completed through the unchanged runner with 334 / 348 PnP inliers and 155 new landmarks
- the one-frame short diagnostic completed with 1 / 1 frames healthy

Validation
- `uv run python -m py_compile src/datasets/kitti.py src/datasets/loader.py scripts/demo_frontend_eth3d.py scripts/diag_pnp_eth3d.py`
- `uv run python -m pytest tests/datasets -q`: 10 passed
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py --profile configs/profiles/kitti_odometry_00.yaml --i1 3 --num_track 1 --out_dir /tmp/kitti_odometry_00_demo_smoke_i1_3`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --profile configs/profiles/kitti_odometry_00.yaml --i1 3 --num_track 1 --scorecard short --threshold_pair_frame_index 9999 --out_dir /tmp/kitti_odometry_00_diag_smoke_i1_3`

Decision
- kept the adapter and runner unchanged
- use bootstrap frame 3 for the first KITTI sequence-00 run

---

## 2026-06-14 — KITTI bootstrap profile defaults

Change
- added KITTI run-profile bootstrap indices `0 -> 3`
- made demo and PnP diagnostic CLI indices optional overrides of profile defaults
- kept the ETH3D fallback at `0 -> 1`

Validation
- dataset and runner compilation passed
- `tests/datasets`: 10 passed
- default KITTI demo selected `0 -> 3` and processed frame 4 successfully
- default KITTI diagnostic completed with 1 / 1 frames healthy

Decision
- kept

---

## 2026-06-14 — Generic runner and dataset wrapper split

Change
- moved dataset-agnostic demo and PnP diagnostic orchestration into `scripts/demo_frontend.py` and `scripts/diag_pnp.py`
- converted the ETH3D entrypoints into thin default-profile wrappers
- added matching KITTI wrappers using the KITTI odometry profile and dataset-specific diagnostic output name
- kept the historical shared helper filename and all SLAM and diagnostic logic unchanged

Validation
- runner compilation passed
- `tests/slam` and `tests/datasets`: 81 passed
- ETH3D demo completed normally
- ETH3D diagnostic completed with 12 / 12 frames healthy
- KITTI demo processed frame 4 successfully
- KITTI diagnostic completed with 1 / 1 frames healthy

Decision
- kept

---

## 2026-06-14 — First longer KITTI sequence-00 diagnosis

Base state
- committed KITTI loader, profile bootstrap `0 -> 3`, and thin KITTI wrappers

Diagnostic step
- ran 30 tracked frames through the KITTI PnP wrapper
- replayed through frame 18 with the existing live-pipeline deep diagnostic

Validation
- `PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 30 --scorecard short --threshold_pair_frame_index 9999 --out_dir /tmp/diag_pnp_kitti_00_longer_30`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 15 --scorecard off --threshold_pair_frame_index 18 --out_dir /tmp/diag_pnp_kitti_00_frame18_deep`

Result
- bootstrap succeeded with 427 landmarks
- frames 4 to 33: 15 accepted and 15 failed
- promotions and local BA succeeded on every frame from 4 to 13
- last clearly healthy frame was 13 with 813 track inliers and 115 / 116 PnP inliers
- first degraded frame was 14
  - strict 8 px PnP failed with 4 inliers
  - 12 px PnP found 52 inliers
  - live rescue accepted 109 / 110 and refreshed the active basis
- first failed frame was 18 from active basis 16
  - 313 track inliers and 56 live PnP correspondences
  - fixed 3 / 5 / 8 / 12 px PnP all failed with zero inliers
  - rescue attempted and failed
  - 53 / 56 live correspondences passed local displacement consistency
  - all 56 live assignments exactly matched the refreshed basis
  - 52 / 56 live landmarks were geometrically drifting over canonical history
- frame 19 rescued once, then frames 20 to 33 failed
- classification: keyframe support-basis geometry quality, not bootstrap weakness, lookup starvation, observation gating, or incoherent 2D tracking

Decision
- kept as diagnosis
- no production change

Next step
- audit the frame-16 rescued pose and support geometry before its active-basis refresh

---

## 2026-06-14 — KITTI frame-16 rescue and support audit

Base state
- KITTI sequence 00 first fails at frame 18 with 0 / 56 live PnP inliers from refreshed basis 16

Diagnostic step
- reproduced frames 13 to 18 with the generic KITTI diagnostic
- used a temporary solver-call interceptor to inspect the exact frame-16 pre-refresh rescue path and support set
- suppressed only the frame-16 support refresh while preserving its accepted pose and observations

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 15 --scorecard off --threshold_pair_frame_index 18 --out_dir /tmp/kitti_frame16_audit_baseline`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=.:scripts uv run python -m py_compile /tmp/diag_kitti_frame16_rescue.py`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=.:scripts uv run python /tmp/diag_kitti_frame16_rescue.py`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 15 --scorecard off --threshold_pair_frame_index 9999 --suppress_support_refresh_frames 16 --out_dir /tmp/kitti_frame16_no_refresh`

Result
- frame 16 strict 8 px failed with zero inliers
- 12 px found 74 / 83 inliers and strict 8 px refit accepted 65 / 83
- 20 px and seeded fallback stages were not reached
- accepted-pose deltas to frames 15 / 17 were small:
  - rotation: 1.68 / 2.04 deg
  - translation direction: 2.78 / 10.04 deg
  - camera-centre direction: 1.67 / 11.36 deg
- accepted support residual median / p90 was 3.29 / 6.69 px
- local 2D support was coherent:
  - 63 / 65 retained
  - residual median / p90: 1.41 / 3.87 px
- refreshed support was spatially concentrated:
  - two occupied coarse cells
  - 96.9 per cent in one component and coarse cell
- 50 / 65 support landmarks already classified as drifting before refresh
- all 56 frame-18 live landmarks were an exact subset of frame-16 accepted support
- suppressing only frame-16 refresh kept frame 16 accepted at 65 / 83 and changed:
  - frame 17 from 43 / 62 without refresh to 64 / 73 with refresh
  - frame 18 from 0 / 56 failed to 49 / 58 accepted

Decision
- diagnosis only
- no production change
- KITTI frame-16 problem is mainly weak support geometry before refresh

Next step
- test one narrow refresh-eligibility guard against spatially concentrated, history-inconsistent rescued support

---

## 2026-06-14 — KITTI frame-19 post-guard diagnosis

Base state
- refresh-only guard committed: blocks rescue refresh when support is concentrated and history-inconsistent, or when support is too weak (fewer than two occupied cells)
- KITTI trusted baseline updated: frame 18 moved from 0 / 56 failed to 49 / 58 accepted, first hard failure moved from frame 18 to frame 19

Diagnostic step
- ran 17 tracked frames (4 through 20) with the KITTI PnP diagnostic
- ran the deep threshold-pair live-pipeline diagnostic on frame 19
- extracted corridor data for frames 16 through 20 from JSONL
- extracted frame-19 support funnel, geometry history, local consistency, and spatial summary from the live pipeline events

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 17 --scorecard short --threshold_pair_frame_index 9999 --out_dir /tmp/kitti_frame19_corridor`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 17 --scorecard off --threshold_pair_frame_index 19 --out_dir /tmp/kitti_frame19_deep`
- JSONL parsing for frames 16 through 20

Result

Corridor after the guard (frames 16 through 20):

| frame | basis before → after | rescue | refresh triggered | n_inliers / n_corr |
|-------|-----------------------|--------|-------------------|---------------------|
| 16 | 14 → 14 | ok | no (concentrated + inconsistent) | 65 / 83 |
| 17 | 14 → 17 | ok | yes (not concentrated) | 64 / 73 |
| 18 | 17 → 17 | ok | no (support too weak, 1 cell) | 49 / 58 |
| 19 | 17 → 17 | fail | — | 0 / 56 |
| 20 | 17 → 17 | ok | no | 40 / 48 |

Frame-19 support funnel (live pipeline, active basis 17):
- raw tracked pairs: 328
- mapped by active lookup: 56
- observation-gated pass: 56
- final PnP correspondences: 56
- 3 / 5 / 8 / 12 px live replay: 0 / 0 / 0 / 0 inliers, `pnp_ransac_failed`
- rescue attempted / succeeded: True / False
- all 56 rescue stages also fail (0 inliers at all thresholds)

Frame-19 support geometry:
- all 56 live landmarks are bootstrap-born (birth_kf = 1, birth_source = bootstrap)
- consistent / drifting: 4 / 52
- full canonical history: 920 observations, median 4.15 px, p90 12.54 px, max 21.63 px
- frame-17 active-basis reprojection of 56 live landmarks: median 5.03 px, p90 9.05 px, max 10.52 px
- local 2D displacement consistency: 54 / 56 pass, median 1.41 px, p90 3.32 px

Frame-19 spatial distribution:
- 47 / 56 correspondences (83.9 %) in one 4 × 3 grid cell
- occupied cells: 5 (current) / 3 (after removing 2 local-consistency outliers)
- `heavily_concentrated: True`
- basis 17 reference view also shows 47 / 56 in the same cell

Frame-18 vs frame-19 comparison:
- both use active basis 17
- frame 18: 49 / 58 rescue succeeds (probably 12 px or 20 px threshold)
- frame 19: 0 / 56 even at all rescue thresholds
- all 56 frame-19 live landmarks exactly match the frame-18 accepted support assignments
- the sharp 49 → 0 inlier drop between the two frames suggests the camera is at a pose where the drifting landmark positions are maximally inconsistent
- frame 20 (same basis) rescues at 40 / 48, confirming the failure at frame 19 is pose-specific rather than a uniform basis blackout

Guard decision analysis:
- frame 16 blocked: concentrated (96.9 % in one cell) AND history-inconsistent → reason `rescued_support_concentrated_history_inconsistent`
- frame 17 allowed: 64 inliers from basis 14 were NOT concentrated (multiple cells) → guard does not evaluate history → refresh to basis 17
- frame 18 blocked: 49 inliers from basis 17 occupy only 1 cell → `support_strong_enough = False` → reason `rescued_support_too_weak`
- frame 17 is the guard miss: its inliers passed the concentration check but the underlying 52 / 56 landmarks are already drifting

Classification
- primary: bad active basis quality — frame-17 rescue basis (64 bootstrap-born, drifting landmarks) was installed because the guard only checks history when support is concentrated; frame 17's spread support bypassed the history check
- secondary: weak support geometry — 52 / 56 live landmarks are geometrically drifting, canonical history p90 12.54 px, basis reproj at 5.03 / 9.05 px before frame 19
- verdict: KITTI frame-19 is mixed (bad active basis quality and weak support geometry)

Decision
- diagnosis only
- no production change

Next step
- determine whether evaluating history-inconsistency for ALL strong rescue support, not only concentrated support, would catch frame-17's drifting basis and prevent the bad-basis installation

---

## 2026-06-14 — History-aware refresh guard extended to all strong support

Base state
- original refresh-only guard committed: blocks refresh when concentrated + history-inconsistent, or support too weak
- KITTI first failure at frame 19 (basis 17 installed by frame 17 rescue; guard missed frame 17 because it was not concentrated)
- ETH3D first failure at frame 19

Hypothesis
- evaluating history-inconsistency for all strong rescued support (not only concentrated) would catch frame-17's drifting basis before installation

Change
- removed the `spatially_concentrated` requirement from the history evaluation condition in `src/slam/frame_pipeline.py`
- history check now runs whenever `support_strong_enough` is True, not only when `support_strong_enough AND spatially_concentrated`
- added new reason string `rescued_support_history_inconsistent` for the spread (non-concentrated) block case
- existing `rescued_support_concentrated_history_inconsistent` reason preserved for the concentrated block case

Validation
- `uv run python -m pytest tests/slam tests/datasets -q`: 81 passed
- KITTI 30-frame diagnostic: ok=16 failed=14, first failure frame 18
- ETH3D 20-frame diagnostic: ok=13 failed=7, first failure frame 15
- KITTI demo (1 frame): completed normally

Result

KITTI corridor (frames 14–20) after patch:

| frame | rescue | refresh | inliers | cells | mcf  | reason |
|-------|--------|---------|---------|-------|------|--------|
| 14    | ok     | BLOCKED | 109     | 5     | 0.74 | history_inconsistent (regression: was allowed) |
| 16    | ok     | BLOCKED | 51      | 4     | 0.92 | concentrated_history_inconsistent (unchanged) |
| 17    | ok     | ALLOWED | 29      | 3     | 0.83 | target case neutralised; 29 inliers < 50 % inconsistent |
| 18    | fail   | —       | 0       | —     | —    | first failure (regression: was frame 19) |

ETH3D corridor:

| frame | rescue | refresh | cells | mcf  | result |
|-------|--------|---------|-------|------|--------|
| 13    | ok     | BLOCKED | 2     | 0.53 | regression: was allowed, caused earlier failure |
| 14    | ok     | BLOCKED | 2     | 0.64 | regression: was allowed |
| 15    | fail   | —       | —     | —    | first failure (regression: was frame 19) |

Root cause of regression
- the history check thresholds (median > 3 px, p90 > 8 px, max > 12 px) are sensitive enough to flag good non-concentrated refreshes as history-inconsistent at early frames
- blocking frame 14 (KITTI) and frames 13–14 (ETH3D) changed the cascade so frame 17's refresh used a smaller, less inconsistent inlier set
- the target guard miss (frame 17 with 64 drifting inliers from basis 14) never materialised because basis 14 was never installed
- the concentration requirement was implicitly filtering out this class of early false-positive history triggers

Decision
- reverted
- production code restored to original guard
- history-inconsistency thresholds need to be calibrated differently for concentrated vs spread support before this extension can be useful

---

## 2026-06-15 - Cross-dataset refresh-history calibration

Base state
- trusted concentration-gated refresh guard
- failed spread-history extension reverted
- KITTI first hard failure at frame 19
- ETH3D first hard failure at frame 19

Diagnostic step
- added a diagnostics-only `rescue_refresh_candidate` JSONL event
- recorded every successful rescue using its final inlier support and pre-refresh landmark history
- replayed KITTI sequence 00 for 30 tracked frames
- replayed ETH3D `cables_2_mono` for 40 tracked frames
- swept per-landmark median, p90, maximum, and drifting-fraction thresholds

Validation
- `uv run python -m py_compile scripts/diag_pnp.py`
- `PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 30 --scorecard off --threshold_pair_frame_index 9999 --out_dir /tmp/refresh_calibration_kitti_current`
- `PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 40 --scorecard off --threshold_pair_frame_index 9999 --out_dir /tmp/refresh_calibration_eth3d_current`

Label rule
- `good_refresh`: the next two frames remain accepted, or existing counterfactual evidence shows blocking worsens survival
- `bad_refresh`: the installed basis remains active into a hard failure within two frames, or existing counterfactual evidence shows blocking improves survival
- `unclear`: neither rule is supported cleanly

Result
- KITTI produced 5 successful rescue candidates; ETH3D produced 9
- recomputed pre-current-frame history matched production history counts on every concentrated KITTI candidate
- the fresh current replay corrected a stale status detail:
  - KITTI frame 18 has two occupied cells and is blocked by concentrated history inconsistency
  - it is not currently blocked by the older one-cell weakness reason
- current history thresholds flagged all 3 labelled bad candidates and 5 of 8 labelled good candidates
- KITTI frame 14 good and frame 17 bad had overlapping pooled history:
  - median 3.63 vs 3.70 px
  - p90 11.76 vs 10.80 px
  - drifting fraction 0.716 vs 0.750
- ETH3D frame 17 unclear and frame 18 bad both had drifting fraction 1.000
- concentration alone or concentration plus current history caught only KITTI frame 16 among the 3 labelled bad candidates
- one sampled higher-threshold rule separated the labelled set:
  - median above 3 px, p90 above 11 px, or maximum above 16 px
  - drifting fraction at least 0.75
- the separation was knife-edge:
  - KITTI frame 17 was exactly 48 / 64 inconsistent landmarks
  - ETH3D frames 17 and 18 differed by one landmark, 17 / 23 versus 18 / 23

Decision
- kept the diagnostics-only event
- production code unchanged
- classification: no robust separator yet
- keep the concentration gate under current thresholds
- next isolate the near-boundary refreshes with single-frame counterfactuals

---

## 2026-06-15 - Single-frame refresh counterfactuals

Base state
- trusted concentration-gated refresh guard
- KITTI and ETH3D first hard failure at frame 19
- near-boundary refresh labels unresolved at KITTI frames 14 / 17 and ETH3D frames 17 / 18

Diagnostic step
- used the existing selected-frame refresh suppression hook
- preserved rescue acceptance, observation append, canonical pose storage, and every non-target refresh
- replayed KITTI frames 4–21 for baseline, frame-14 suppression, and frame-17 suppression
- replayed ETH3D frames 2–21 for baseline, frame-17 suppression, and frame-18 suppression

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 18 --scorecard off --threshold_pair_frame_index 9999 --out_dir /tmp/single_refresh_kitti_baseline`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 18 --scorecard off --threshold_pair_frame_index 9999 --suppress_support_refresh_frames 14 --out_dir /tmp/single_refresh_kitti_no14`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 18 --scorecard off --threshold_pair_frame_index 9999 --suppress_support_refresh_frames 17 --out_dir /tmp/single_refresh_kitti_no17`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 20 --scorecard off --threshold_pair_frame_index 9999 --out_dir /tmp/single_refresh_eth3d_baseline`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 20 --scorecard off --threshold_pair_frame_index 9999 --suppress_support_refresh_frames 17 --out_dir /tmp/single_refresh_eth3d_no17`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 20 --scorecard off --threshold_pair_frame_index 9999 --suppress_support_refresh_frames 18 --out_dir /tmp/single_refresh_eth3d_no18`

Result
- KITTI baseline first failed at frame 19; frame 20 recovered and frame 21 failed
- suppressing KITTI frame 14 kept its 109 / 110 rescue but moved first failure to frame 18
- suppressing KITTI frame 17 kept its 64 / 73 rescue and first failure at frame 19, but weakened frame 18 to 23 / 62 and removed the frame-20 recovery
- ETH3D baseline first failed at frame 19 and stayed failed through frame 21
- suppressing ETH3D frame 17 kept its 23 / 23 rescue but moved first failure to frame 18
- suppressing ETH3D frame 18 kept its 23 / 23 rescue and left the frame-19-to-21 failure corridor unchanged

Classification
- KITTI frame 14: `load_bearing_good_refresh`
- KITTI frame 17: `load_bearing_good_refresh`
- ETH3D frame 17: `load_bearing_good_refresh`
- ETH3D frame 18: `mostly_neutral_refresh`
- overall: `single-frame counterfactuals reveal usable refresh labels`

Decision
- kept as diagnosis
- production code unchanged
- current status sharpened with causal refresh labels
- next compare causal labels for a feature beyond pooled history thresholds

---

## 2026-06-15 - Downstream reuse comparison across labelled refresh events

Base state
- four causal refresh labels: KITTI 14/17 and ETH3D 17 as load_bearing_good_refresh; ETH3D 18 as mostly_neutral_refresh
- pooled history thresholds confirmed not robust (ETH3D 17 vs 18 differ by one landmark)
- question: what at-refresh feature separates the three good refreshes from the neutral one

Diagnostic step
- ran diag_pnp_kitti.py with --num_track 20 and diag_pnp_eth3d.py with --num_track 22
- parsed frame_summary and rescue_refresh_candidate events from JSONL
- recorded at-refresh properties (n_inliers, cells, max_cell_frac, drifting_frac) and downstream frame outcomes (f+1, f+2, f+3: ok, inliers, same_basis) for each target refresh frame

Validation
- `uv run python scripts/diag_pnp_kitti.py --num_track 20 --out_dir /tmp/diag_refresh_kitti --scorecard off`
- `uv run python scripts/diag_pnp_eth3d.py --num_track 22 --out_dir /tmp/diag_refresh_eth3d --scorecard off`
- KITTI summary: frames=20, ok=16, failed=4, rescue=9/5, refresh=2
- ETH3D summary: frames=22, ok=17, failed=5, rescue=10/9, refresh=9

At-refresh properties
- KITTI 14: 109 inliers, 5 cells, max_cell_frac=0.743, drifting_frac=0.716, basis 13→14
- KITTI 17: 64 inliers, 5 cells, max_cell_frac=0.844, drifting_frac=0.750, basis 14→17
- ETH3D 17: 23 inliers, 3 cells, max_cell_frac=0.565, drifting_frac=1.000, basis 16→17
- ETH3D 18: 23 inliers, 3 cells, max_cell_frac=0.652, drifting_frac=1.000, basis 17→18
- all four cleared the guard (support_strong_enough=True, spatially_concentrated=False)

Downstream persistence (f+1, f+2, f+3 — ok / inliers / same_basis)
- KITTI 14 (good): True/62/same  True/65/same  True/64/same  → 3/3 accepted
- KITTI 17 (good): True/49/same  False/0/same  True/40/same  → 2/3 accepted
- ETH3D 17 (good): True/23/same  False/0/new-basis  False/0/new-basis  → 1/3 accepted
- ETH3D 18 (neutral): False/0/same  False/0/same  False/0/same  → 0/3 accepted
- (ETH3D f+2 for frame-17 uses basis-18 because frame-18 triggers another refresh)

Result
- downstream reuse is a clean monotone separator: good refreshes ≥ 1 accepted downstream frame; neutral = 0
- limiting pair ETH3D 17 (good, 1/3) vs ETH3D 18 (neutral, 0/3) is at-refresh-indistinguishable:
  - both have 23 inliers, 3 cells, drifting_frac=1.000, landmark obs-count median 15–16
  - largest_component_fraction=1.000 for both; no spatial metric separates them
- downstream reuse is strictly retrospective: it cannot be measured at refresh time

Classification
- separator identified: `downstream reuse` (≥1 accepted frame in f+1..f+3)
- type: retrospective only — no current at-refresh signal predicts it for ETH3D 17 vs 18

Decision
- kept as diagnosis
- production code unchanged
- current status updated with downstream reuse finding and limiting-pair ambiguity
- next investigate why basis-18 fails at frame 19 while basis-17 allowed frame 18: per-landmark geometry comparison (depth, triangulation baseline, viewpoint angle) between the two bases

---

## 2026-06-15 - ETH3D basis-17 versus basis-18 geometry

Base state
- ETH3D frame 17 is a load-bearing refresh; frame 18 is mostly neutral
- both refreshes have 23 inliers, 3 occupied cells, and drifting fraction 1.000
- current refresh-time summaries do not separate them

Diagnostic step
- added `scripts/diag_eth3d_basis17_18_geometry.py`
- replayed the trusted ETH3D baseline through frame 19
- snapshotted the installed bases after refreshes 17 and 18
- compared set membership, landmark provenance, observation counts, depth, 3D extent, singular values, ray spread, viewpoint angles, and baseline/depth ratios
- tracked each frozen basis independently into frame 19
- scored both live correspondence sets under a common frame-19 pose propagated from local ETH3D ground-truth motion

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python -m py_compile scripts/diag_eth3d_basis17_18_geometry.py`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_eth3d_basis17_18_geometry.py --output /tmp/eth3d_basis17_18_geometry.json`

Result
- both bases contain the exact same 23 landmark IDs; intersection 23 / 23, Jaccard 1.000
- both contain 16 bootstrap and 7 map-growth landmarks with the same birth-frame split
- frame 18 only contributes one additional observation per landmark
- centred 3D singular values are identical at 17.262 / 3.787 / 2.484
- basis-18 depth spread is only slightly larger: median 21.63 versus 22.79, coefficient of variation 0.083 versus 0.074
- basis-18 ray and historical viewpoint spread are slightly broader, not narrower
- both installed bases contain the same three near-collocated landmark pairs
- at frame 19, basis 17 yields 18 correspondences and basis 18 yields 22; all 18 basis-17 live landmarks are contained in basis 18
- both PnP attempts fail
- common-reference residual median / p90 is 6.07 / 14.65 px for basis 17 and 6.05 / 12.98 px for basis 18
- basis 18 has slightly better DLT and pose-Jacobian conditioning
- the four basis-18-only live correspondences are not uniformly harmful: their comparative residuals are 1.93, 7.07, 7.70, and 11.58 px
- the propagated reference validates to 2.08° rotation and 21.0° translation-direction disagreement over frame 17 to 18, so residuals are comparative rather than absolute

Classification
- `basis-17 and basis-18 still lack a clear geometric separator`

Decision
- kept as diagnosis
- production code unchanged
- current status materially sharpened
- next isolate frame-19 correspondence compatibility and minimal-sample pose dispersion on the 18 shared live landmarks

---

## 2026-06-15 - ETH3D frame-16 rescue acceptance-path audit

Base state
- ETH3D first failure at frame 19, driven by geometry drift from bad canonical poses at frames 12 and 16
- frame 16 accepted as a 20 px localisation-only rescue; accepted pose is far worse than the local temporal interpolation
- open question: which stage accepted the bad pose and why did it pass all gates

Diagnostic step
- added `scripts/diag_eth3d_frame16_rescue_stages.py`
- replayed through frame 19 to retrieve state, then re-ran all frame-16 rescue stages against the live basis-15 feature set (28 correspondences)
- audited: stage-by-stage inlier counts, spatial gate stats, cheirality, temporal gate margins, strict-on-loose fraction
- compared accepted pose against a time-weighted frame-15-to-17 interpolation on the same 28 correspondences

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_eth3d_frame16_rescue_stages.py --out /tmp/frame16_rescue_stages.json`

Result

Stage-by-stage:

| stage | threshold px | loose_ok | n_loose | strict-on-loose |
|-------|-------------|----------|---------|-----------------|
| strict_8px | 8 | False | 0 | — |
| stage1_12px | 12 | False | 0 (RANSAC failed) | — |
| stage2_20px | 20 | True | 24 | 0 / 24 = 0 % |
| stage2_seed_40px | 40 | True | 28 | — |

- 12 px RANSAC found zero inliers from 28 correspondences; stage2_20px is the first stage to produce a pose
- strict-on-loose fraction = 0 %: all 24 loose inliers fail 8 px strict refit
- temporal gate: translation direction delta 8.07 deg, camera centre direction delta 5.84 deg; both trivially under the 120 deg threshold
- interpolated pose on corr16: median 6.21 px, p90 8.49 px, 6 / 28 above 8 px
- accepted 20 px pose: 0 / 28 at strict 8 px; interpolation is 71.30 % lower squared error
- no better candidate was available under the current rescue stages; the acceptance was locally consistent with all current gates
- strict-on-loose fraction = 0 is a new detectable quality signal; the 120 deg temporal gate does not detect this class of incoherence

Classification: `mixed`
- no better candidate available under current rescue stages
- accepted candidate locally passes all gates but is globally harmful
- root contamination is upstream (frame-12 bad canonical pose); frame 16 perpetuates it

Decision
- diagnosis only
- production code unchanged
- current status sharpened with stage audit, strict-on-loose fraction, and temporal gate margin details

Next step
- add a strict-on-loose fraction gate to `_accept_loose_localisation_fallback`; require a minimum fraction of 20 px inliers to also survive 8 px strict refit before accepting as localisation-only rescue

---

## 2026-06-15 - ETH3D frame-19 shared-18 consensus fragility

Base state
- ETH3D frame-19 fails PnP with 22 live correspondences and zero accepted inliers at all thresholds
- basis 17 yields 18 live correspondences; basis 18 yields 22, containing all 18 plus four extra (IDs 181, 226, 360, 588)
- both PnP attempts fail; question is whether the shared-18 core is already fragile or whether the four extras drive the failure

Diagnostic step
- added `scripts/diag_eth3d_frame19_shared18.py`
- replayed trusted ETH3D baseline through frame 18, built frame-19 correspondences with `build_pnp_correspondences_with_stats`
- split 22 correspondences into shared-18 (IDs not in {181, 226, 360, 588}) and extra-4
- Part A: ran `_inlier_sweep` at 8 / 12 / 20 / 40 px and `_hypothesis_sample_analysis` (2000 DLT minimal samples) on shared-18
- Part B: leave-one-out — removed each shared landmark in turn, ran sweep on 17-point residual set
- Part C: added each extra landmark individually to shared-18, ran sweep on the 19-point set

Validation
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python -m py_compile scripts/diag_eth3d_frame19_shared18.py`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_eth3d_frame19_shared18.py --output /tmp/eth3d_frame19_shared18.json`

Result

Part A — shared-18 baseline:
- best inliers: 0 / 0 / 2 / 11 at 8 / 12 / 20 / 40 px
- 1157 of 2000 DLT minimal samples produced valid models
- pairwise rotation dispersion: median 130.5°, p90 172.0°, max 180.0°
- no valid hypothesis achieves more than 0 inliers at 8 px or more than 1 inlier at 20 px
- the 18 shared correspondences scatter every minimal-sample pose across the full rotation manifold

Part B — leave-one-out on shared-18:
- at 8 px: no single removal improves above 1 inlier (8 removals each achieve 1)
- at 12 px: no single removal improves above 1 inlier (13 removals each achieve 1)
- at 20 px: no single removal improves above 2 inliers
- at 40 px: removing lm 311 uniquely achieves 17 / 17 inliers (all remaining landmarks fit); next best is 12
- lm 311 reference residual is 10.25 px; removing it does not cure strict-threshold failure

Part C — add extras one at a time to shared-18:
- adding lm 181 (1.93 px ref-residual): 2 at 40 px — worsens from 11
- adding lm 226 (11.58 px ref-residual): 6 at 40 px — worsens from 11
- adding lm 360 (7.07 px ref-residual): 7 at 40 px — worsens from 11
- adding lm 588 (7.70 px ref-residual): 19 at 40 px — all 19 fit at 40 px; best individual extra
- full 22 together: 2 at 40 px — catastrophic combined interference
- at 8 / 12 / 20 px all extras give at most 1 / 3 / 3 inliers regardless

Classification
- primary: `shared_set_already_consensus_fragile` at all PnP-relevant thresholds (8 / 12 / 20 px)
- secondary: lm 311 is a partial outlier visible only at 40 px; the four extras interact destructively at 40 px
- overall: `mixed` — shared-18 is the dominant failure path; extras add interference but do not uniquely cause the collapse

Decision
- diagnosis only
- production code unchanged
- current status sharpened with shared-18 dispersion and LOO findings
- next step is frame-16 rescue-pose audit to understand why a bad canonical pose was accepted and how it propagated into the landmark geometry

---

## 2026-06-15 - Cross-dataset strict-on-loose calibration

Base state
- ETH3D frame-16 rescue-stages audit identified strict-on-loose = 0/24 as a detectable quality signal
- current_status.md proposed: add a strict-on-loose gate to `_accept_loose_localisation_fallback`
- open question: is strict-on-loose == 0 a broadly useful pathological signal, or just a one-off?

Diagnostic step
- added `scripts/diag_strict_on_loose_calibration.py`
- replayed ETH3D cables_2_mono (22 frames) and KITTI sequence 00 (20 frames)
- extracted from `pose_out["stats"]` for every accepted rescue:
  - `pnp_support_rescue_loose_threshold_px`
  - `pnp_support_rescue_loose_inliers`
  - `pnp_support_rescue_subset_count`
  - `pnp_support_rescue_subset_strict_inliers`
  - `pnp_support_rescue_loose_localisation_fallback_succeeded`
- strict-on-loose fraction = subset_strict_inliers / subset_count
- labelled each event with known refresh labels from single-frame counterfactuals

Result

| dataset | frame | thr | loose | n_inliers | sol_count | sol_frac | fallback | refresh_label |
|---------|-------|-----|-------|-----------|-----------|----------|----------|---------------|
| ETH3D | 8 | 20 | 90 | 90 | 90/90 | 1.000 | no | unclear |
| ETH3D | 10 | 12 | 69 | 56 | 56/69 | 0.812 | no | unclear |
| ETH3D | 12 | 20 | 40 | 40 | 0/40 | 0.000 | YES | unclear (bad canonical pose) |
| ETH3D | 13 | 20 | 36 | 36 | 0/36 | 0.000 | YES | unclear |
| ETH3D | 14 | 20 | 32 | 32 | 0/32 | 0.000 | YES | unclear |
| ETH3D | 15 | 20 | 30 | 30 | 0/30 | 0.000 | YES | unclear |
| ETH3D | 16 | 20 | 24 | 24 | 0/24 | 0.000 | YES | load-bearing refresh, bad canonical pose |
| ETH3D | 17 | 20 | 23 | 23 | 0/23 | 0.000 | YES | load_bearing_good_refresh |
| ETH3D | 18 | 20 | 23 | 23 | 0/23 | 0.000 | YES | mostly_neutral_refresh |
| KITTI | 14 | 12 | 52 | 109 | 52/52 | 1.000 | no | load_bearing_good_refresh |
| KITTI | 16 | 12 | 74 | 65 | 65/74 | 0.878 | no | refresh_blocked_guard |
| KITTI | 17 | 20 | 64 | 64 | 0/64 | 0.000 | YES | load_bearing_good_refresh |
| KITTI | 18 | 20 | 49 | 49 | 0/49 | 0.000 | YES | refresh_blocked_guard |
| KITTI | 20 | 20 | 40 | 40 | 0/40 | 0.000 | YES | refresh_blocked_guard |

Key findings:
- strict-on-loose = 0 is the normal state for 10 of 14 accepted rescues across both datasets
- strict-on-loose = 0 applies equally to known-good refreshes (ETH3D 17, KITTI 17) and bad poses (ETH3D 12, 16)
- ETH3D frame 17 (load_bearing_good_refresh): sol = 0/23 — identical to ETH3D frame 16 (bad pose): sol = 0/24
- KITTI frame 17 (load_bearing_good_refresh): sol = 0/64 — identical pattern to bad cases
- the transition to sol = 0 at ETH3D frame 12 and KITTI frame 17 is the signature of the late-pipeline regime where geometry has drifted sufficiently that even 12 px RANSAC fails
- within the late-pipeline 20 px fallback class, strict-on-loose = 0 for all events; there is no discrimination
- a gate requiring sol > 0 for fallback acceptance would block all ETH3D frames 12–18 and KITTI frames 17–20, including all known-good refreshes in that window
- blocking ETH3D frame 17 refresh was already shown by counterfactual to move first failure from frame 19 to frame 18

Classification
- `strict-on-loose is too noisy to trust`

Decision
- no production change
- strict-on-loose gate proposal (from frame-16 rescue audit) is RETRACTED
- current_status.md updated: classification changed, best next step updated
- next question: what signal distinguishes bad canonical pose frames (ETH3D 12, 16) from other 20 px fallback frames in the same class

---

## 2026-06-16 - Bad-vs-useful late fallback comparison

Base state
- trusted baseline after strict-on-loose retraction
- strict-on-loose = 0 is treated as a late 20 px fallback regime marker, not a harmfulness discriminator

Diagnostic step
- added `scripts/diag_bad_vs_useful_fallbacks.py`
- replayed ETH3D through frame 22 and KITTI through frame 22
- compared bad canonical-pose fallbacks against load-bearing useful fallbacks:
  - bad: ETH3D frames 12 and 16
  - useful: ETH3D frame 17 and KITTI frame 17
  - neutral references: ETH3D frame 18, KITTI frames 18 and 20
- measured event table fields, local pose-neighbour deltas, neighbour-interpolation residual changes, exact accepted-support history, next-frame residuals, support overlap, and downstream reuse

Validation
- `uv run python -m py_compile scripts/diag_bad_vs_useful_fallbacks.py`
- `uv run python scripts/diag_bad_vs_useful_fallbacks.py --eth3d_stop_frame 22 --kitti_stop_frame 22 --out /tmp/bad_vs_useful_fallbacks.json --quiet`
- parsed `/tmp/bad_vs_useful_fallbacks.json`

Result
- all chosen frames are 20 px localisation-only fallbacks:
  - ETH3D 12: basis 10→12, 40 / 45 inliers, refresh allowed, bad canonical pose
  - ETH3D 16: basis 15→16, 24 / 28 inliers, refresh allowed, bad canonical pose
  - ETH3D 17: basis 16→17, 23 / 23 inliers, refresh allowed, load-bearing good refresh
  - KITTI 17: basis 14→17, 64 / 73 inliers, refresh allowed, load-bearing good refresh
  - KITTI 18 / 20: 49 / 58 and 40 / 48 inliers, refresh blocked by concentrated history inconsistency
- pose-deviation evidence separates the main bad/useful set:
  - ETH3D 12 / 16 rotation-path excess: 8.97° / 9.94°
  - ETH3D 17 / KITTI 17 rotation-path excess: 2.49° / 0.76°
  - neighbour interpolation reduces accepted-support squared error for ETH3D 12 / 16 by 28.2 % / 70.5 %
  - the same interpolation worsens accepted-support squared error for ETH3D 17 / KITTI 17 by 473.5 % / 260.2 %
- raw camera-centre turn / outside-chord is not safe on its own:
  - KITTI 17 is useful but still has a large centre-turn / outside-chord artefact
- forward support viability does not separate harmful pose from useful fallback:
  - future accepted-inlier reuse fractions are 0.900 / 0.958 for bad ETH3D 12 / 16
  - useful ETH3D 17 / KITTI 17 are 1.000 / 0.781
- history residuals on accepted support do not separate:
  - ETH3D 12 prior history p90 is 5.61 px with only 5 % inconsistent landmarks
  - useful ETH3D 17 / KITTI 17 prior p90 is 10.87 / 10.80 px
- neutral references remain cautionary:
  - KITTI 18 has low rotation-path excess but neighbour interpolation greatly improves residuals; refresh is blocked
  - ETH3D 18 remains mostly neutral over the tested horizon

Classification
- `pose_deviation_from_local_motion`
- the clean signal is rotation-path excess plus accepted-support residual comparison against neighbour interpolation
- neighbour interpolation is retrospective and not directly usable as an at-rescue-time production gate

Decision
- kept as diagnosis
- production code unchanged
- current status updated
- next step: calibrate a previous-motion-only proxy for the pose-deviation oracle before considering any gate

---

## 2026-06-17 - Previous-motion-only proxy calibration

Base state
- trusted baseline after the bad-vs-useful late fallback comparison
- retrospective pose-deviation oracle separates bad ETH3D frames 12 / 16 from useful ETH3D frame 17 and KITTI frame 17
- open question: whether a previous-motion-only proxy can approximate that oracle online

Diagnostic step
- extended `scripts/diag_bad_vs_useful_fallbacks.py` with a diagnostic-only `previous_motion_proxy` block
- replayed ETH3D and KITTI through frame 22
- compared accepted rescue pose against:
  - previous accepted pose delta
  - previous camera-centre motion turn
  - one-sided rotation-path excess from the previous two accepted poses
  - constant-velocity extrapolated pose from the previous two accepted poses
  - accepted-support residuals under the past extrapolated pose

Validation
- `uv run python -m py_compile scripts/diag_bad_vs_useful_fallbacks.py`
- `uv run python scripts/diag_bad_vs_useful_fallbacks.py --eth3d_stop_frame 22 --kitti_stop_frame 22 --out /tmp/bad_vs_useful_previous_motion.json --quiet`
- parsed `/tmp/bad_vs_useful_previous_motion.json`

Result
- event table reproduced the labelled set:
  - ETH3D 12: basis 10→12, 40 / 45 inliers, 20 px localisation-only, bad canonical pose
  - ETH3D 16: basis 15→16, 24 / 28 inliers, 20 px localisation-only, bad canonical pose
  - ETH3D 17: basis 16→17, 23 / 23 inliers, 20 px localisation-only, load-bearing good refresh
  - KITTI 17: basis 14→17, 64 / 73 inliers, 20 px localisation-only, load-bearing good refresh
  - ETH3D 18 and KITTI 18 / 20 stayed useful neutral references
- previous-pose delta does not separate: ETH3D 16 and ETH3D 17 have similar previous-pose rotation / centre-direction deltas
- previous-motion turn does not separate: ETH3D 17 is 135.68° because the bad frame-16 pose is already in the past
- one-sided rotation-path excess does not separate: ETH3D 17 is 9.94°, higher than both bad frames under the past-only definition
- past extrapolated residuals do not reproduce the retrospective oracle:
  - ETH3D 12 / 16 previous-motion residual reductions: -266.9 % / -2.5 %
  - ETH3D 17 / KITTI 17 previous-motion residual reductions: -2532.1 % / -223.2 %
  - the extrapolated pose is worse than the accepted pose for both bad and useful main frames
- neutral KITTI 18 is an added caution: past extrapolation improves its support residuals by 75.2 % despite low retrospective rotation-path excess and blocked refresh
- retrospective neighbour interpolation remains the clean signal:
  - bad ETH3D 12 / 16 rotation-path excess: 8.97° / 9.94°
  - useful ETH3D 17 / KITTI 17 rotation-path excess: 2.49° / 0.76°
  - bad interpolation residual reductions: 28.2 % / 70.5 %
  - useful interpolation residual reductions: -473.5 % / -260.2 %

Classification
- `one-sided motion proxy is too weak`

Decision
- kept as diagnosis
- production code unchanged
- current status updated
- do not promote a pure previous-motion-only guard

---

## 2026-06-20 — Rescue-time support-conditioned signal comparison

Base state
- trusted baseline after previous-motion-only proxy calibration
- retrospective pose-deviation oracle (rotation-path excess + neighbour interpolation residual reduction) cleanly separates bad from useful
- previous-motion-only proxies are too weak: they are contaminated by the bad frame-16 pose being in the motion history
- next clean question: is there a signal available at rescue time from the accepted rescue candidate and its support set?

Clarification of strict-on-loose = 0
- `pnp_support_rescue_subset_strict_inliers = 0` for all 20 px fallback rescues is a code-path artefact
- the loose localisation fallback branch returns at `pnp_frontend.py` line 518 before the strict RANSAC is attempted
- strict-on-loose = 0 is always 0 for this code path; it is not a measure of post-refinement inlier quality

Diagnostic step
- new script: `scripts/diag_rescue_candidate_quality.py`
- replayed ETH3D (frames 2–20) and KITTI (frames 2–21) under the current guard
- for each target frame, extracted the accepted inlier set from `pnp_inlier_mask` in `pose_out` after nonlinear refinement
- computed post-refinement residuals of accepted inliers under the accepted pose
- computed fine threshold retention: fractions at or below 10 / 12 / 14 / 16 px
- computed DLT condition (sigma_min / sigma_max from inlier set)
- attempted bootstrap subsample stability (50 random 6-point subsets)

Event table

| dataset | frame | stage | n_inliers | median_px | p90_px | frac_≤12px | frac_>14px | frac_>16px | label |
|---------|-------|-------|-----------|-----------|--------|------------|------------|------------|-------|
| ETH3D | 12 | loose_20px | 40 | 14.66 | 17.44 | 0.175 | 0.550 | 0.300 | bad canonical pose |
| ETH3D | 16 | loose_20px | 24 | 10.71 | 17.36 | 0.667 | 0.250 | 0.208 | bad canonical pose |
| ETH3D | 17 | loose_20px | 23 | 4.05 | 9.24 | 0.957 | 0.000 | 0.000 | load-bearing good refresh |
| KITTI | 17 | loose_20px | 64 | 5.03 | 9.06 | 0.984 | 0.016 | 0.016 | load-bearing good refresh |
| ETH3D | 18 | loose_20px | 23 | 5.97 | 11.18 | 0.957 | 0.043 | 0.043 | neutral |
| KITTI | 18 | loose_20px | 49 | 16.98 | 18.86 | 0.041 | 0.959 | 0.694 | neutral (blocked by guard) |
| KITTI | 20 | loose_20px | 40 | 2.06 | 4.19 | 1.000 | 0.000 | 0.000 | neutral (blocked by guard) |

Signal comparison

- residual median: min(bad) 10.71 px vs max(good) 5.03 px — clean separation
- frac_at_most_12px: max(bad) 0.667 vs min(good) 0.957 — clean separation
- frac_above_14px: min(bad) 0.250 vs max(good) 0.016 — clean separation
- frac_above_16px: min(bad) 0.208 vs max(good) 0.016 — clean separation
- DLT condition (sigma_min / sigma_max): ETH3D cases all ~2e-5 regardless of label — does not separate
- bootstrap stability: 6-point DLT subsets numerically ill-conditioned; all 50 trials fail for every frame — unavailable

Mechanism
- bad pose: 20 px RANSAC finds a consensus at the wrong camera location; nonlinear refinement converges to a local minimum keeping inlier residuals at 10–15 px
- good pose: 20 px RANSAC finds a consensus at the correct camera location (landmarks have drifted); refinement from the right region gives inlier residuals of 4–6 px despite landmark drift
- the post-refinement inlier residual distribution is a direct quality indicator of whether nonlinear refinement found a geometrically meaningful solution

Neutral reference notes
- ETH3D 18 (neutral): median 5.97 px, frac_≤12px 0.957 — indistinguishable from good frames; signal would correctly not block it
- KITTI 18 (neutral, already blocked by guard): median 16.98 px — agrees with the guard; consistent not redundant
- KITTI 20 (neutral, blocked by guard for spatial concentration): median 2.06 px — excellent refined pose; signal would correctly not block it; guard's spatial reasoning is independent

Classification
- `residual-shape signal looks promising`

Decision
- kept as diagnosis
- new diagnostic script committed
- current status updated
- experiment log updated

---

## 2026-06-20 — Residual-shape action selection

Base state
- residual-shape signal confirmed: post-refinement accepted-inlier median cleanly separates bad canonical-pose frames (10–15 px) from good rescue frames (4–6 px)
- open question: what should the signal control — canonical pose storage, refresh, both, or unclear?

Diagnostic step
- read existing counterfactual evidence (no new code run)
- mapped three distinct control call sites in `frame_pipeline.py`:
  - `append_tracked_observations_to_seed` (lines 840–851): observation append
  - `_refresh_active_lookup_basis_from_rescued_support` (lines 890–905): active basis refresh
  - `store_current_pose` (line 968): canonical pose storage in `seed["poses"][current_kf]`
- evaluated four hypothetical actions against every labelled frame using existing counterfactuals and the interpolation oracle

Action matrix (from existing evidence)

| action | ETH3D 12/16 (bad pose, load-bearing refresh) | ETH3D 17 / KITTI 17 (good refresh) | neutral references |
|--------|-----------------------------------------------|------------------------------------|--------------------|
| A1: keep both (current) | frame-19 fails; kf=12/16 dominate landmark history | no harm | no harm |
| A2: keep refresh, no canonical pose | hypothetical: refresh keeps support continuity; bad observations absent from history | equivalent to A1 (not flagged) | equivalent to A1 |
| A3: keep canonical pose, block refresh | tested: first failure earlier (−2 to −3 frames); harmful | tested: first failure earlier; harmful | neutral to harmful |
| A4: reject entirely | untested; worse than A3 | worse than A3 | worse |

Evidence for A3 being harmful
- suppress frame-12+16 refresh: first failure 19 → 16
- suppress frame-16 only: first failure 19 → 17
- suppress frame-17 (ETH3D): first failure 19 → 18

Evidence for canonical pose as harm carrier
- kf=16 interpolation oracle: replacing bad canonical pose reduces kf=16 squared error by 71%; full-history p90 drops from 10.87 to 8.48 px
- kf=12 contributes 56% of frames 15–18 squared error; kf=16 contributes 25% of 340-obs squared error
- both contributions flow solely from the bad pose being stored in `seed["poses"][12/16]` and then evaluated against appended observations

Coupling constraint identified
- canonical pose agreement invariant (added 2026-06-12) requires `seed["poses"][kf]` to agree with the active keyframe record's pose
- A2 cannot simply skip `store_current_pose` while keeping the refresh; requires either a synthetic/interpolated canonical pose, invariant relaxation for rescue-flagged frames, or a different active-basis kf

Classification
- `canonical_pose_storage looks like the right control point`

Decision
- kept as diagnosis
- production code unchanged
- current status updated with action matrix and coupling constraint
- next step: diagnostic counterfactual that replaces (not skips) the bad canonical pose at frame 16 with the constant-velocity extrapolated pose from frames 14–15, then measures effect on frame-19 landmark history

---

## 2026-06-20 — Canonical pose storage counterfactual (ETH3D frame 16)

Base state
- first failure frame 19; ETH3D frames 12 and 16 identified as bad canonical-pose outliers
- previous session classified `canonical_pose_storage` as the right control point from oracle evidence

Hypothesis
- replacing the stored canonical pose at kf=16 with a constant-velocity extrapolated pose from kf=14→15 should reduce landmark-history residuals, validating canonical pose storage as the harm carrier

Method
- baseline: unmodified replay through frame 19 (no production changes)
- counterfactual: analytical only — replace kf=16 residuals in the 340-row history with residuals under the past-extrap pose; no seed mutation
- script: `scripts/diag_canonical_pose_storage.py`
- extrapolation: `_extrapolate_pose(pose_14, pose_15, alpha)` with `alpha = (ts_16 - ts_14) / (ts_15 - ts_14) = 2.000` (near-uniform frame spacing)
- oracle comparison: time-interpolated pose from kf=15→17 with `alpha = 0.500`

Frame-16 local results (22 live frame-19 landmarks, kf=16 bundle)

| pose | median_px | p90_px | above_8 | sq_error | sq_reduction |
|------|-----------|--------|---------|----------|--------------|
| accepted rescue (bad) | 10.97 | 17.51 | 16/22 | 3 220 | — |
| past-extrap (kf14→15, α=2.000) | 10.73 | 15.43 | 19/22 | 2 774 | −13.9% |
| oracle interp (kf15→17, α=0.500) | 5.95 | 8.16 | 4/22 | 924 | −71.3% |

Pose delta vs accepted rescue pose

| pose | rotation_delta_deg | camera_centre_distance |
|------|--------------------|----------------------|
| past-extrap | 4.76° | 3.17 |
| oracle | 6.27° | 2.90 |

Full-history counterfactual results (340 rows, 22 live landmarks)

| configuration | median_px | p90_px | above_8 | sq_error | sq_reduction |
|--------------|-----------|--------|---------|----------|--------------|
| baseline | 2.87 | 10.87 | 52 | 12 662 | — |
| past-extrap kf16 | 2.87 | 10.67 | 55 (+3) | 12 215 | −3.5% |
| oracle kf16 | 2.70 | 8.48 | 40 (−12) | 10 366 | −18.1% |

Findings
- past-extrap median barely improves (10.97 → 10.73 px, +2.2%) and WORSENS above-8 count (16 → 19)
- past-extrap gives only 3.5% full-history sq_error reduction — below noise threshold for "helps weakly" (5%)
- the past-extrap pose is a DIFFERENT bad pose, not a good one; alpha=2.000 overshoots the true frame-16 position in the wrong direction
- oracle confirms that canonical pose storage IS load-bearing (71.3% kf=16 bundle reduction; 18.1% global reduction)
- root cause of extrap failure: constant-velocity model from kf=14→15 is insufficient; both 14 and 15 are themselves 20px rescues; accumulated extrapolation error places the predicted pose far from the true camera position at frame 16

Classification
- `canonical pose storage is strongly implicated` (oracle evidence: 18.1% global sq_error reduction with a correct pose at kf=16)
- `past-motion-only extrapolation is not a viable rescue-time substitute` (3.5% global reduction, worse above-8; median 10.73 px vs target <8 px)

Decision
- kept as diagnosis
- production code unchanged
- canonical pose storage confirmed as the harm carrier; the rescue-time proxy challenge is now the open sub-problem
- next step: explore alternative synthetic canonical poses at rescue time: (a) store previous-keyframe pose at kf=16 rather than the rescue pose; (b) suppress the kf=16 canonical pose from history with a relaxed invariant for rescue-flagged frames; (c) investigate whether the residual-shape signal can gate canonical-pose acceptance independently of the refresh decision

---

## 2026-06-20 — Canonical pose proxy comparison (ETH3D frame 16)

Base state
- canonical pose storage confirmed as harm carrier; past-motion extrapolation found not viable
- next question: which rescue-time proxy poses are viable substitutes for the bad 20px canonical pose?

Method
- script: `scripts/diag_frame16_proxy_poses.py`
- replay through frame 19 (no production changes); capture frame-16 PnP correspondences and loose inlier mask from `pose_out["corrs"]` and `pose_out["pnp_inlier_mask"]`
- evaluate five proxy candidates on the 22 live-frame-19 landmarks' kf=16 observations and full 340-row history replacement

Candidates tested

| candidate | family | rescue-time only? |
|-----------|--------|------------------|
| frame-15 canonical pose | temporal prior | yes |
| 40px RANSAC on full 28 correspondences | full-support re-solve | yes |
| trimmed 50% refit (best 16 of 24 loose inliers, 12px) | trimmed-support refit | yes |
| pruned <15px refit (19 of 24 loose inliers, 12px) | pruned-support refit | yes |
| oracle kf15→17 interpolation | retrospective reference | no |

Frame-16 local results (22 live frame-19 landmarks, kf=16 bundle)

| candidate | median_px | p90_px | above_8 | sq_error | local_sq_red |
|-----------|-----------|--------|---------|----------|-------------|
| accepted bad rescue | 10.97 | 17.51 | 16/22 | 3 220 | 0% |
| frame-15 pose | 8.82 | 15.81 | 12/22 | 2 430 | 24.5% |
| 40px re-solve | 3.23 | 7.45 | 2/22 | 464 | 85.6% |
| trimmed 50% refit | FAILED | — | — | — | — |
| pruned <15px refit | FAILED | — | — | — | — |
| oracle interp | 5.95 | 8.16 | 4/22 | 924 | 71.3% |

40px re-solve inlier counts on full 28 correspondences: 25/28 at 8px, 28/28 at 12px, 28/28 at 20px

Full-history counterfactual (340 rows, 22 live landmarks)

| candidate | median_px | p90_px | above_8 | sq_error | sq_reduction | Δabove_8 |
|-----------|-----------|--------|---------|----------|-------------|---------|
| baseline | 2.87 | 10.87 | 52 | 12 662 | — | — |
| frame-15 pose | 2.87 | 10.50 | 48 | 11 872 | 6.2% | −4 |
| 40px re-solve | 2.62 | 8.48 | 38 | 9 906 | 21.8% | −14 |
| oracle | 2.70 | 8.48 | 40 | 10 366 | 18.1% | −12 |

Pose delta (40px re-solve vs accepted bad rescue)
- rotation_delta_deg: 6.75°
- translation_direction_delta_deg: 8.63°
- camera_centre_distance: 2.98

Findings
- the 40px RANSAC on the full 28 rescue correspondences finds the TRUE camera pose at kf=16 (3.23 px median, BETTER than the oracle's 5.95 px)
- 25/28 correspondences agree with the 40px result at strict 8px — this is the correct pose the strict RANSAC failed to find
- the trimmed and pruned refits both fail entirely: RANSAC at 12px on the best 12–19 loose inliers under the bad pose finds 0 valid poses, confirming that sorting inliers by residual under the bad pose selects the support most biased toward the wrong minimum
- frame-15 pose gives only 8.82 px (24.5% local sq_error reduction) and still has 12/22 above-8px — weak improvement
- root cause of bad rescue: stage2_20px (5000 trials) converges to a wrong local minimum with 24 inliers; the seeded 40px path in the existing rescue pipeline would reach the correct minimum, but it is never executed because stage2_20px succeeds first and returns early

Classification
- `robust re-refinement looks promising` — specifically, a wider-threshold RANSAC on the FULL rescue correspondence set escapes the 20px wrong local minimum
- frame-15 pose: `helps only weakly` (8.82 px, 24.5% local sq_error reduction)
- trimmed/pruned refits: `not viable` (fail completely)

Decision
- kept as diagnosis
- production code unchanged
- the 40px re-solve is the strongest rescue-time proxy found; next step is stability testing across RANSAC seeds and across both bad frames (12 and 16), before any production patch

---

## 2026-06-20 — 40px re-solve seed stability and cross-frame safety test

Base state
- 40px re-solve on ETH3D frame 16 found correct pose (3.23px) from previous session
- question: is this stable across seeds and safe for good/neutral frames?

Method
- script: `scripts/diag_40px_resolve_stability.py`
- replay ETH3D (frames 2–19) and KITTI (frames 2–21); capture rescue correspondences for each target frame
- run 8 RANSAC seeds per frame at 40px threshold, 5000 trials each
- evaluate on full rescue correspondences; for bad frames (ETH3D 12, 16) also evaluate on live-19 bundle and full-history replacement

Frames tested:
- ETH3D 12 (bad canonical pose, 45 corrs)
- ETH3D 16 (bad canonical pose, 28 corrs)
- ETH3D 17 (good refresh, 23 corrs)
- KITTI 17 (good refresh, 73 corrs)
- KITTI 18 (neutral, 58 corrs)
- KITTI 20 (neutral, 48 corrs)

Results

| frame | accepted median | 8px inliers | 40px best median | 40px 8px inliers | success | tight | max rot delta |
|-------|-----------------|-------------|-----------------|-----------------|---------|-------|---------------|
| ETH3D 12 (bad) | 15.25 | 0/45 | 2.13 | 45/45 | 8/8 | YES | 0.000° |
| ETH3D 16 (bad) | 11.39 | 6/28 | 3.75 | 25/28 | 8/8 | NO | 5.19° |
| ETH3D 17 (good) | 4.05 | 18/23 | 3.46 | 19/23 | 6/8 | NO | 5.25° |
| KITTI 17 (good) | 5.52 | 52/73 | 6.05 | 49/73 | 8/8 | YES | 0.26° |
| KITTI 18 (neutral/bad) | 17.38 | 2/58 | 4.06 | 48/58 | 5/8 | NO | 4.22° |
| KITTI 20 (neutral) | 2.22 | 40/48 | 3.10 | 40/48 | 8/8 | NO | 6.13° |

Local sq_error reduction (full corrs):
- ETH3D 12: 96.1%; ETH3D 16: 87.6%; KITTI 18: 72.8%
- ETH3D 17: 13.4%; KITTI 17: −70.3% (expected: strict-8px pose already good; 40px LM gets more outlier noise); KITTI 20: 5.4%

Full-history replacement (live-19 set, bad frames):
- ETH3D 12: sq_red=35.9%, Δabove_8=−22; ETH3D 16: sq_red=21.8%, Δabove_8=−14

ETH3D 12 40px inlier detail: all 8 seeds produce IDENTICAL result (max_rot_delta=0.000°, max_cc=0.000); correct pose gives 45/45 at 8px — the only viable 40px local minimum for this frame

ETH3D 16 40px cluster: seed=0 finds correct pose (25/28 at 8px); some seeds find alternative bad local minima (up to 31.91px median); 8px inlier count cleanly discriminates (25 vs <<10)

KITTI 17 best 40px: 1.24° from accepted, 0.11 camera-centre distance — functionally same pose; cluster perfectly tight

KITTI 18 surprise: accepted median=17.38px (2/58 at 8px) — third bad rescue pose (previously labelled neutral); 40px re-solve gives 4.06px, 48/58 at 8px

Good/neutral frames would not be gated (accepted medians: ETH3D 17=4.05px, KITTI 17=5.52px, KITTI 20=2.22px — all well below 8px threshold); their best 40px results differ from accepted by ≤1.25°

Stability pattern
- HIGH correspondence count + dominant correct minimum (ETH3D 12: 45 corrs): perfectly stable, seed-independent
- LOWER correspondence count or competing local minima (ETH3D 16: 28, KITTI 18: 58, KITTI 20: 48): cluster not tight; seed-dependent
- Key discriminator: the correct local minimum always has far more 8px inliers than the alternatives; selecting by highest 8px inlier count on full corrs gives reliable recovery

Classification
- `40 px re-solve is stable and promising`
- safe condition: apply only when rescue-residual-shape gate triggers (accepted inlier median > ~8px); select result with highest 8px inlier count from a small multi-seed run (3–5 seeds)

Decision
- kept as diagnosis; production code unchanged
- stability confirmed on both ETH3D bad frames; benign on all tested good/neutral frames
- next step: gated multi-seed 40px re-solve patch with 8px-inlier-count selection criterion; test on full sequences before production merge

---

## 2026-06-20 — Gated 40 px shadow integration test

Base state
- 40 px full-set re-solve looked promising on labelled bad and good rescue frames
- open question: whether a residual-gated multi-seed 40 px shadow path remains beneficial and safe over longer real sequence replays before any production patch

Method
- new script: `scripts/diag_40px_shadow_integration.py`
- replayed ETH3D `cables_2_mono` for 40 tracked frames and KITTI sequence 00 for 30 tracked frames
- left the real pipeline pose, refresh, observation append, canonical pose storage, and seed state unchanged
- for each accepted rescue, computed accepted-inlier residual median under the accepted pose
- triggered the shadow path only when that median was above 8 px
- when triggered, ran 5 seeds of 40 px RANSAC, 5000 trials each, on the full rescue correspondence set
- selected the candidate with the highest strict 8 px inlier count on the full set, breaking ties by lower full-set residual median
- estimated current-frame history impact by evaluating stored frame observations under the accepted pose versus the selected shadow pose

Validation
- `uv run python -m py_compile scripts/diag_40px_shadow_integration.py`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_40px_shadow_integration.py --out /tmp/diag_40px_shadow_integration.json`
- parsed `/tmp/diag_40px_shadow_integration.json`

ETH3D result
- baseline replay unchanged: 17 / 40 ok, first failure frame 19, 9 successful rescues, 9 refreshes
- rescue frames considered: 8, 10, 12, 13, 14, 15, 16, 17, 18
- gate fired only on frames 12 and 16
- frame 12: accepted median / p90 15.25 / 20.28 px; shadow 2.13 / 4.95 px; strict 8 px 0 / 45 -> 45 / 45; seeds 5 / 5; local verdict better
- frame 16: accepted median / p90 11.39 / 20.68 px; shadow 3.75 / 7.84 px; strict 8 px 6 / 28 -> 25 / 28; seeds 5 / 5; local verdict better
- known useful / neutral frames did not trigger: frame 17 support median 4.05 px, frame 18 support median 5.97 px
- history estimates: frame 12 stored-observation sq-error reduction 95.9% and above-8 reduction 40; frame 16 sq-error reduction 84.7% and above-8 reduction 16
- downstream reuse estimate: frame 12 support reused as inlier support for frames 13-15; frame 16 support reused at frames 17-18 and remained pose-eligible at failed frame 19

KITTI result
- baseline replay unchanged: 16 / 30 ok, first failure frame 19, 5 successful rescues, 2 refreshes
- rescue frames considered: 14, 16, 17, 18, 20
- gate fired only on frame 18
- frame 18: accepted median / p90 17.38 / 23.87 px; shadow 4.06 / 18.35 px; strict 8 px 2 / 58 -> 48 / 58; seeds 3 / 5; local verdict better
- known useful / neutral frames did not trigger: frame 14 support median 1.93 px, frame 17 support median 5.03 px, frame 20 support median 2.06 px
- history estimate: frame 18 stored-observation sq-error reduction 90.0% and above-8 reduction 43
- downstream reuse estimate: frame 18 support stayed pose-eligible at frames 19-21 and was reused as inlier support at frame 20

Classification
- `gated 40 px re-solve is promising and broadly safe`

Decision
- kept as diagnosis
- production behaviour unchanged
- current status updated because the trusted interpretation materially sharpened from labelled-frame stability to sequence-level viability
- clean next step, if production work resumes: a narrow canonical-pose-storage proxy patch gated by accepted-inlier residual median, leaving refresh and rescue acceptance untouched

---

## 2026-06-20 - Canonical-pose proxy production patch

Base state
- trusted baseline kept rescue acceptance, rescue refresh, append-on-rescue, BA, and keyframe policy unchanged
- diagnostic shadow path showed that accepted-inlier residual median above 8 px identified the bad canonical-pose rescue frames

Change
- added a production canonical-pose proxy path in `src/slam/frame_pipeline.py`
- trigger: successful loose rescue with accepted-inlier residual median above 8 px
- action: run five 40 px RANSAC re-solves on the full rescue correspondence set and select by highest strict 8 px inlier count, then lower full-set residual median
- storage: replace only `store_current_pose` input for the current keyframe when a proxy is selected
- reporting: passed the new canonical proxy audit fields through standard frame summaries
- rescue acceptance, refresh, observation append, BA, map growth, and keyframe policy were not changed

Validation
- `cd /home/samio/WSLGit/Robotics/Robotics/MonoSLAM && UV_CACHE_DIR=/tmp/uv-cache uv run python -m pytest tests/slam tests/datasets -q`: 81 passed
- `cd /home/samio/WSLGit/Robotics/Robotics/MonoSLAM && UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_eth3d.py --num_track 22 --scorecard short --threshold_pair_frame_index 9999 --out_dir /tmp/diag_pnp_eth3d_after_canonical_proxy_patch`
- `cd /home/samio/WSLGit/Robotics/Robotics/MonoSLAM && UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/diag_pnp_kitti.py --num_track 22 --scorecard short --threshold_pair_frame_index 9999 --out_dir /tmp/diag_pnp_kitti_after_canonical_proxy_patch`
- `cd /home/samio/WSLGit/Robotics/Robotics/MonoSLAM && UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py`
- `cd /home/samio/WSLGit/Robotics/Robotics/MonoSLAM && UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python scripts/demo_frontend_kitti.py --num_track 1 --out_dir /tmp/kitti_demo_after_canonical_proxy_patch`

Result
- ETH3D summary: 17 / 22 ok, first failure frame 19, rescue 10 / 9, refresh 9
- KITTI summary: 16 / 22 ok, first failure frame 19, rescue 11 / 5, refresh 2
- ETH3D frame 12: trigger fired, proxy stored, 5 seeds, selected 45 / 45 strict 8 px, selected median 2.13 px
- ETH3D frame 16: trigger fired, proxy stored, 5 seeds, selected 25 / 28 strict 8 px, selected median 3.75 px
- ETH3D frames 17 and 18 did not trigger
- KITTI frame 18: trigger fired, proxy stored, 5 seeds, selected 48 / 58 strict 8 px, selected median 4.06 px
- KITTI frames 17 and 20 did not trigger
- frame 19 failure remains in both diagnostics, so the short-horizon downstream survival did not improve

Decision
- kept as a narrow canonical-pose storage patch
- classification: `result inconclusive`
