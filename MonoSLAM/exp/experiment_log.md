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
