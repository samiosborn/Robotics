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
