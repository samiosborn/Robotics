# Current trusted baseline

## Kept changes
- monotonic viable-consensus selection
- stronger second-stage 20 px rescue search budget
- seeded second-stage 20 px fallback from 40 px loose support
- append-on-rescue for existing landmark observations
- no stale-pruning during rescue-frame append
- pose-eligible linked-landmark guard for keyframe promotion
- earlier rescued-support refresh
- minimal local bundle adjustment on promoted non-rescue keyframes
- canonical pose agreement enforced in keyframe reads
- live-pipeline threshold diagnostics in `scripts/diag_pnp_eth3d.py`
- live-pipeline local displacement-consistency diagnostics
- live-pipeline refreshed-basis assignment diagnostics
- diagnostic-only selected rescue-refresh suppression replay
- diagnostic-only frame-16 accepted-pose quality audit

## Reverted or failed experiments
- proactive rescued-basis retracking before late support collapse
- low-cardinality multi-seed retry in second-stage seeded 40 px rescue
- other narrow support-refresh timing variants that did not improve behaviour enough to keep

## Current long-run behaviour
- BA-enabled pipeline with promotion guard and earlier rescued-support refresh now stays healthy through frame 18
- first current failure is frame 19
- long-run summary at 40 tracked frames:
  - ok / failed: 17 / 23
  - promoted frames: 2, 4, 6
  - rescue attempted / succeeded: 10 / 9
  - support refresh triggered frames: 8, 10, 12, 13, 14, 15, 16, 17, 18
  - BA attempted / succeeded: 3 / 3

## Current first failure
- frame 19
- active keyframe: 18 refreshed basis
- track inliers: 616
- raw tracked pairs: 616
- mapped by active lookup: 22
- valid landmarks / valid X_w: 22 / 22
- observation-gated pass: 22
- final PnP correspondences: 22
- pipeline result: `pnp_ransac_failed`
- rescue attempted / succeeded: true / false

## Current interpretation
- frame 19 live-pipeline diagnostics are now trustworthy
- live diagnostics use the refreshed frame-18 active basis, not the stale promoted-keyframe reference
- fixed live-bundle replay at frame 19 confirms:
  - 22 live correspondences
  - 3 px: 1 / fail
  - 5 px: 1 / fail
  - 8 px: 1 / fail
  - 12 px: 1 / fail
- live 8/12 threshold-pair replay rejects all 22 correspondences
- all 22 live correspondences pass the frame-18-to-19 local displacement-consistency check
  - median residual: 2.34 px
  - p90 residual: 4.72 px
- each fixed-threshold replay produces 902 valid PnP models from 1000 trials, but no model exceeds one inlier
- current bottleneck is no longer lookup starvation or stale diagnostic bundle selection
- current bottleneck is not low-cardinality search failure or incoherent 2D tracking
- frame-19 refreshed-basis assignment audit confirms:
  - frame 18 installed 23 feature-to-landmark pairs from exactly 23 accepted support pairs
  - no installed pair was outside accepted support
  - no accepted pair was missing from the installed lookup
  - no duplicate landmark reuse, missing landmark ids, observation mismatches, or assignment conflicts
  - all 22 frame-19 live correspondences are unique feature and landmark assignments
  - all 22 exactly match the preserved refreshed lookup and exact frame-18 observations
  - the live set is a 22-pair subset of the 23-pair refreshed basis
  - birth sources are 15 bootstrap landmarks and 7 map-growth landmarks
- frame-18 reprojection of the 22 live linked landmarks is already loose:
  - median: 6.19 px
  - p90: 11.25 px
  - maximum: 17.35 px
  - 14 / 22 are within 8 px
- full-history audit of the exact 22 live landmarks confirms broad geometry drift:
  - 340 valid observations across the 22 landmarks
  - every landmark has at least 10 valid observations
  - pooled full-history median / p90 / maximum: 2.87 / 10.87 / 19.16 px
  - all 22 landmarks have either p90 above 8 px or maximum above 12 px
  - bootstrap vs map-growth median / p90: 3.04 / 10.50 px vs 2.41 / 12.03 px
  - latest-BA vs outside-latest-BA median / p90: 2.99 / 10.59 px vs 2.32 / 11.85 px
  - no birth-source, birth-frame, or BA-participation subgroup explains the failure
- residual growth is synchronised by canonical observation frame:
  - frame 12: median 15.05 px, p90 17.40 px, 0 / 22 within 8 px
  - frame 16: median 10.97 px, p90 17.51 px, 6 / 22 within 8 px
  - frame 18: median 6.19 px, p90 11.25 px, 14 / 22 within 8 px
  - frame-12 bootstrap vs map-growth medians are 14.94 px vs 15.16 px
  - frame-16 bootstrap vs map-growth medians are 10.58 px vs 11.88 px
- canonical rescue-pose audit confirms two sharp pose-history outliers:
  - frames 10 and 12–18 are localisation-only rescues; frame 11 is normal
  - every rescue in this corridor refreshes the active support basis
  - every accepted frame 10–18 is stored as a canonical pose
  - no local BA runs in the corridor
  - frame 12 vs frames 11 / 13:
    - rotation deltas: 4.88 / 7.43 deg
    - translation-direction deltas: 37.48 / 20.11 deg
    - camera-centre direction deltas: 32.87 / 12.71 deg
    - camera-motion turn: 118.99 deg
    - adjacent camera-centre step ratio: 3.01
    - rotation-path excess: 8.97 deg
  - frame 16 vs frames 15 / 17:
    - rotation deltas: 6.42 / 6.46 deg
    - translation-direction deltas: 8.07 / 10.80 deg
    - camera-centre direction deltas: 5.84 / 6.52 deg
    - camera-motion turn: 135.68 deg
    - frame 16 lies behind the frame-15-to-17 camera-centre chord
    - rotation-path excess: 9.94 deg
  - every other evaluable frame 10–17 has rotation-path excess at or below 4.25 deg
- frames 12 and 16 disproportionately dominate the exact 22-landmark history:
  - 44 / 340 observations, or 12.94 per cent
  - 62.25 per cent of pooled squared reprojection error
  - 73.08 per cent of residuals above 8 px
  - excluding both frames reduces pooled median / p90 from 2.87 / 10.87 px to 2.31 / 6.19 px
  - excluding both changes the landmark classification from 22 / 22 drifting to 8 / 22 drifting
  - frame 13 immediately returns to median / p90 1.95 / 5.76 px
  - frame 17 returns to median / p90 4.13 / 9.28 px
- the broad history inconsistency is therefore mainly driven by bad canonical rescue poses at frames 12 and 16, not by smooth distributed drift
- a smaller later tail remains at frames 17 and 18, so the two bad poses do not explain every part of the frame-19 failure
- rescue poses do not directly update landmark positions because rescue frames skip map growth and local BA
- the remaining causal question is whether refreshing the active support basis from those bad poses propagates the later frame-19 support failure
- the failure is now classified as coherent 2D tracks with internally consistent assignments attached to a geometrically incompatible 3D support set
- frame-19 geometry drift is broad rather than concentrated in a landmark subgroup
- shared-18 consensus fragility (2026-06-15):
  - the 18 correspondences shared by both live bases are already consensus-fragile at all PnP-relevant thresholds
  - best inliers at 8 / 12 / 20 / 40 px: 0 / 0 / 2 / 11
  - 1157 / 2000 DLT minimal samples produce valid models; pairwise rotation dispersion median 130.5°, p90 172.0°, max 180.0°
  - the hypothesis distribution covers the full rotation manifold — complete geometric incoherence
  - no single leave-one-out removal improves above 1 inlier at 8 or 12 px; no shared correspondence is the unique poison
  - lm 311 (ref-residual 10.25 px) is a 40 px outlier: removing it gives 17 / 17 at 40 px, but still 0 at 8 px
  - adding each of the four extra (basis-18-only) correspondences individually: lm 588 improves 40 px to 19 / 19; lm 181 / 226 / 360 each worsen it; combining all four collapses to 2 at 40 px
  - the extras interact destructively in combination but do not uniquely cause the strict-threshold failure
  - classification: `mixed` — shared-18 is the dominant failure path; lm 311 and the extras add secondary interference at 40 px

## Rescue-refresh counterfactual
- one 40-frame replay kept rescue localisation unchanged but suppressed active-basis refresh at frames 12 and 16
- frame 12 remained the same accepted rescue:
  - 45 final PnP correspondences
  - 40 inliers
  - active basis retained at frame 10 instead of moving to frame 12
- frame 13 and frame 14 still rescued successfully and refreshed normally
- frame 15 rescued with 18 / 34 inliers, but its support was too weak to refresh
- frame 16 then failed with 0 / 22 inliers, so the requested frame-16 refresh suppression was never reached
- first failure moved earlier from frame 19 to frame 16
- 40-frame ok / failed changed from 17 / 23 to 14 / 26
- counterfactual frame 19 used 18 live landmarks from active basis 14:
  - pooled canonical-history median / p90: 1.89 / 11.82 px
  - 14 / 18 landmarks still classified as drifting
  - frame 12 contributed 41.20 per cent of pooled squared reprojection error
  - frame 12 contributed 56.00 per cent of residuals above 8 px
- the nine-landmark intersection with the baseline frame-19 set remained broad:
  - baseline median / p90: 2.63 / 10.42 px
  - counterfactual median / p90: 1.98 / 11.02 px
  - drifting landmarks changed from 9 / 9 to 7 / 9
- suppressing frame-12 refresh therefore reduced some median residuals but materially worsened support survival
- bad rescue poses are correlated with canonical-history error, but active-basis refresh is not their main harmful propagation path
- frame-12 refresh is beneficial for support continuity despite the bad canonical pose

## Frame-16-only rescue-refresh counterfactual
- a fresh trusted baseline replay reproduced:
  - first failure at frame 19
  - 17 / 40 frames ok
  - 10 / 9 rescues attempted / succeeded
  - nine support refreshes
- one 40-frame replay preserved frame-12 refresh and suppressed only the frame-16 active-basis refresh
- frame 16 remained the same accepted rescue:
  - 28 final PnP correspondences
  - 24 inliers
  - active basis retained at frame 15 instead of moving to frame 16
- frame 17 then failed with 0 / 16 inliers
- frames 18 and 19 also failed with 0 / 19 and 0 / 22 inliers
- first failure moved earlier from frame 19 to frame 17
- 40-frame ok / failed changed from 17 / 23 to 15 / 25
- rescue attempted / succeeded changed from 10 / 9 to 10 / 7
- counterfactual frame 19 used 22 live landmarks from active basis 15:
  - pooled canonical-history median / p90: 2.27 / 10.71 px
  - 21 / 22 landmarks remained classified as drifting
  - frame 16 contributed 24.62 per cent of pooled squared reprojection error
  - frame 16 contributed 35.00 per cent of residuals above 8 px
- the 16-landmark intersection with the baseline frame-19 set remained broad:
  - baseline median / p90: 2.79 / 11.00 px
  - counterfactual median / p90: 2.27 / 11.04 px
  - all 16 landmarks remained classified as drifting in both runs
- the lower counterfactual median does not indicate recovery because frames 17 and 18 failed and added no accepted canonical observations
- frame-16 refresh supports downstream continuity rather than causing the frame-19 collapse
- frame-16 refresh is not the main problem

## Frame-16 accepted rescue-pose quality audit
- a fresh replay through frame 19 reproduced:
  - 17 accepted frames and first failure at frame 19
  - frame 16 accepted as a 20 px localisation-only rescue with 24 / 28 inliers
  - the exact same 22 unique frame-19 live landmarks as the trusted live assignment audit
- the active basis before frame 16 was canonical frame 15, so there was no distinct older active-basis pose to explain the jump
- frame 16 remains a sharp neighbour-pose outlier:
  - rotation delta to frames 15 / 17 / 18: 6.42 / 6.46 / 7.19 deg
  - translation-direction delta: 8.07 / 10.80 / 13.00 deg
  - camera-centre direction delta: 5.84 / 6.52 / 6.83 deg
  - frame-15-to-16 and frame-16-to-17 camera-motion turn: 135.68 deg
  - frame 16 lies behind the frame-15-to-17 camera-centre chord with projection alpha -0.179
  - camera path ratio: 2.15
  - rotation-path excess: 9.94 deg
- all 22 live landmarks have one clean observation at each of frames 15–18
- canonical residuals for the same 22 landmarks are:
  - frame 15 median / p90: 3.09 / 6.10 px, 0 / 22 above 8 px
  - frame 16 median / p90: 10.97 / 17.51 px, 16 / 22 above 8 px
  - frame 17 median / p90: 4.13 / 9.28 px, 5 / 22 above 8 px
  - frame 18 median / p90: 6.19 / 11.25 px, 8 / 22 above 8 px
- within frames 15–18, frame 16 contributes:
  - 56.34 per cent of squared reprojection error
  - 55.17 per cent of residuals above 8 px
- across the full 340-observation history, frame 16 alone contributes:
  - 25.43 per cent of squared reprojection error
  - 30.77 per cent of residuals above 8 px
- a time-weighted frame-15-to-17 pose interpolation at frame 16 materially improves the same observations:
  - median / p90: 5.95 / 8.16 px
  - 4 / 22 above 8 px
  - 71.30 per cent lower frame-16 squared error
- replacing only frame 16 with that interpolated pose in the full history changes:
  - pooled median / p90: 2.87 / 10.87 px to 2.70 / 8.48 px
  - residuals above 8 px: 52 to 40
  - pooled squared error: 18.13 per cent lower
- frame 16 is therefore intrinsically a bad accepted rescue pose relative to its neighbours and the later live landmark set
- correcting frame 16 would materially reduce the geometry-drift story, but would not remove the independent frame-12 outlier or the smaller frame-17/18 tail
- classification: frame-16 accepted rescue pose is a main outlier

## Current open question
Why does the frame-16 20 px localisation-only rescue accept a pose that is far worse than the local frame-15-to-17 temporal interpolation on the same later-live landmarks?

## Best next step
Keep rescue refresh enabled and diagnostically audit the frame-16 rescue candidate poses and acceptance stages against the local temporal reference.

## KITTI sequence-00 frame-16 diagnosis
- pre-guard baseline still fails first at frame 18 with 0 / 56 live PnP inliers from refreshed basis 16
- frame 16 is not a loose 20 px pose acceptance:
  - strict 8 px failed with zero inliers
  - 12 px found 74 / 83 inliers
  - strict 8 px refit on that subset accepted 65 / 83
  - 20 px and seeded fallback stages were not reached
- the accepted frame-16 pose is locally credible:
  - rotation delta to frames 15 / 17: 1.68 / 2.04 deg
  - translation-direction delta: 2.78 / 10.04 deg
  - camera-centre direction delta: 1.67 / 11.36 deg
  - frame-16 support residual median / p90: 3.29 / 6.69 px
  - frame-15-to-17 interpolation is substantially worse on the same support
- the accepted support is 2D-coherent but geometrically weak as a refreshed basis:
  - 63 / 65 pass local displacement consistency
  - local residual median / p90: 1.41 / 3.87 px
  - 96.9 per cent lies in one image component and coarse cell
  - only two coarse cells are occupied
  - 50 / 65 landmarks already classify as drifting before refresh
- all 56 frame-18 live landmarks are an exact subset of the 65 frame-16 refreshed-support landmarks
- suppressing only frame-16 refresh keeps the same accepted frame-16 pose but removes the hard failure:
  - frame 17 improves from 43 / 62 to 64 / 73 and refreshes basis 17
  - frame 18 improves from 0 / 56 failed to 49 / 58 accepted and refreshes basis 18
- classification: KITTI frame-16 problem is mainly weak support geometry before refresh

## KITTI refresh-only guard result
- narrow guard committed: blocks rescue refresh when support is spatially concentrated and history-inconsistent, or when occupied cells is below two
- guard effect on the 17-frame corridor (frames 4 through 20):
  - frame 16: rescue ok at 65 / 83, refresh BLOCKED (concentrated + history-inconsistent)
  - frame 17: rescue ok at 64 / 73, refresh ALLOWED (not concentrated, basis → 17)
  - frame 18: rescue ok at 49 / 58, refresh BLOCKED (two cells, max-cell fraction 0.918, concentrated + history-inconsistent)
  - frame 19: rescue FAILS at 0 / 56 from basis 17
  - frame 20: rescue ok at 40 / 48 from basis 17, refresh BLOCKED (two cells, max-cell fraction 0.975, concentrated + history-inconsistent)

## KITTI frame-19 post-guard diagnosis
- new first hard failure: frame 19, active basis 17, 0 / 56 inliers, rescue also fails
- frame-19 support funnel (live, from basis 17):
  - raw tracked pairs: 328
  - mapped by active lookup: 56
  - observation-gated pass: 56
  - final PnP correspondences: 56
  - all four live replays (3 / 5 / 8 / 12 px) and all rescue stages: 0 inliers
- frame-19 support geometry:
  - all 56 live landmarks are bootstrap-born (birth_kf = 1)
  - consistent / drifting: 4 / 52
  - canonical history: 920 obs, median 4.15 px, p90 12.54 px, max 21.63 px
  - frame-17 basis reprojection of 56 live landmarks: median 5.03 px, p90 9.05 px
  - local 2D displacement consistency: 54 / 56 pass, median 1.41 px, p90 3.32 px
- frame-19 spatial distribution:
  - 47 / 56 (83.9 %) in one 4 × 3 grid cell
  - `heavily_concentrated: True`
  - basis-17 reference view shows the same concentration
- frame-18 vs frame-19:
  - both frames use active basis 17
  - frame 18: 49 / 58 rescue succeeds; all 56 frame-19 live landmarks are exactly the frame-18 accepted support
  - frame 19: 0 / 56 even at all rescue thresholds; frame 20 recovers at 40 / 48 from the same basis
  - the 49 → 0 drop is pose-specific: the camera at frame 19 is at a position where the geometry error in the 52 drifting landmarks is worst-case for RANSAC

## KITTI guard miss at frame 17
- frame 17 passed the guard because its 64 inliers from basis 14 were not concentrated
- the guard only evaluates history-inconsistency when the support is already spatially concentrated
- frame 17's spread inliers bypassed the history check: the guard did not detect that 52 / 56 of those landmarks are geometry-drifting bootstrap points
- the resulting basis-17 installation is the proximate cause of the frame-19 failure

## KITTI frame-19 classification
- primary: bad active basis quality — basis 17 (rescue, 64 drifting bootstrap landmarks) was installed because the guard's history check is concentration-gated
- secondary: weak support geometry — 52 / 56 live landmarks geometrically drifting, canonical p90 12.54 px, basis reproj at 5.03 / 9.05 px before frame 19
- verdict: KITTI frame-19 is mixed (bad active basis quality and weak support geometry)

## Refresh-history calibration across KITTI and ETH3D
- a fresh current-guard replay collected every successful rescue over the practical horizons:
  - KITTI sequence 00: 30 tracked frames, 16 accepted, 14 failed, 5 successful rescues, 2 refreshes
  - ETH3D `cables_2_mono`: 40 tracked frames, 17 accepted, 23 failed, 9 successful rescues, 9 refreshes
- candidate labels use a short explicit horizon:
  - `good_refresh`: the next two frames remain accepted, or existing counterfactual evidence shows that blocking the refresh worsens survival
  - `bad_refresh`: the installed basis remains active into a hard failure within two frames, or existing counterfactual evidence shows that blocking improves survival
  - `unclear`: neither rule is supported cleanly
- known good and bad KITTI spread support overlaps strongly under the current history rule:
  - frame 14 good refresh: max-cell fraction 0.743, pooled median / p90 / max 3.63 / 11.76 / 23.17 px, drifting fraction 0.716
  - frame 17 bad refresh: max-cell fraction 0.844, pooled median / p90 / max 3.70 / 10.80 / 21.63 px, drifting fraction 0.750
- ETH3D shows the same late overlap:
  - frame 16 good refresh: pooled median / p90 / max 2.03 / 7.41 / 18.35 px, drifting fraction 0.875
  - frame 17 is unclear after one healthy hand-off frame: 2.33 / 10.87 / 19.16 px, drifting fraction 1.000
  - frame 18 bad refresh: 2.51 / 10.61 / 19.16 px, drifting fraction 1.000
- the current history rule catches all three labelled bad candidates but also flags five of eight labelled good candidates
- concentration alone and concentration plus current history avoid labelled-good false positives, but each catches only KITTI frame 16 and misses spread bad candidates at KITTI frame 17 and ETH3D frame 18
- one more conservative sampled history rule separated the current labelled set:
  - per-landmark median above 3 px, p90 above 11 px, or maximum above 16 px
  - candidate drifting fraction at least 0.75
- that separator is not robust:
  - KITTI frame 17 sits exactly at 48 / 64, or 0.750
  - ETH3D frame 17 and frame 18 differ by one landmark at 17 / 23 versus 18 / 23
  - a one-landmark classification change would remove the separation
- pooled median, p90, maximum, and current drifting fraction therefore do not provide a robust monotonic separator across both datasets
- classification: no robust separator yet

## Single-frame refresh counterfactual labels
- focused replays through frame 21 suppressed one otherwise-allowed refresh while preserving rescue acceptance, observation append, canonical pose storage, and all other refresh decisions
- KITTI frame 14 is a load-bearing good refresh:
  - frame 14 still rescued at 109 / 110 when refresh was suppressed
  - basis 13 was retained instead of installing basis 14
  - frame 17 still rescued and refreshed, but with only 29 / 58 inliers instead of 64 / 73
  - first failure moved earlier from frame 19 to frame 18
- KITTI frame 17 is a load-bearing good refresh:
  - frame 17 still rescued at 64 / 73 when refresh was suppressed
  - basis 14 was retained until frame 18 rescued at 23 / 62 and installed basis 18
  - first failure stayed at frame 19, but the baseline frame-20 recovery disappeared and frames 19–21 all failed
- ETH3D frame 17 is a load-bearing good refresh:
  - frame 17 still rescued at 23 / 23 when refresh was suppressed
  - basis 16 was retained instead of installing basis 17
  - frame 18 failed at 0 / 18, moving first failure earlier from frame 19 to frame 18
- ETH3D frame 18 is mostly neutral over the tested horizon:
  - frame 18 still rescued at 23 / 23 when refresh was suppressed
  - basis 17 was retained instead of installing basis 18
  - frame 19 still failed, and frames 20–21 also remained failed
- the earlier proxy label for ETH3D frame 18 as bad is not supported causally: suppressing its refresh does not improve survival
- single-frame counterfactuals reveal usable refresh labels, but the labels do not yet imply a robust observable separator

## Downstream reuse comparison (2026-06-15)
- ran KITTI (num_track=20) and ETH3D (num_track=22) through diag_pnp scripts; parsed frame_summary and rescue_refresh_candidate JSONL events
- at-refresh properties for all four labelled events:
  - KITTI 14: 109 inliers, 5 cells, max_cell_frac=0.743, drifting_frac=0.716, basis 13→14
  - KITTI 17: 64 inliers, 5 cells, max_cell_frac=0.844, drifting_frac=0.750, basis 14→17
  - ETH3D 17: 23 inliers, 3 cells, max_cell_frac=0.565, drifting_frac=1.000, basis 16→17
  - ETH3D 18: 23 inliers, 3 cells, max_cell_frac=0.652, drifting_frac=1.000, basis 17→18
  - all four: support_strong_enough=True, spatially_concentrated=False
- downstream persistence (ok / inliers) at f+1, f+2, f+3:
  - KITTI 14 (good): T/62, T/65, T/64 → 3/3 accepted
  - KITTI 17 (good): T/49, F/0, T/40 → 2/3 accepted
  - ETH3D 17 (good): T/23, F/0†, F/0 → 1/3 accepted
  - ETH3D 18 (neutral): F/0, F/0, F/0 → 0/3 accepted
  - †ETH3D frame 17 f+2 uses basis-18 (frame 18 triggers its own refresh)
- separator: downstream reuse (n_ok_in_window ≥ 1) cleanly separates all three good from the neutral
- limiting pair ETH3D 17 vs 18 is at-refresh-indistinguishable: identical inlier count (23), cells (3), drifting_frac (1.000), and obs-count median (15 vs 16)
- downstream reuse is strictly retrospective; no current at-refresh feature predicts it for the ETH3D boundary
- classification: `downstream reuse` is the separator

## ETH3D basis-17 versus basis-18 geometry (2026-06-15)
- added a diagnostics-only replay that snapshots both installed bases and tracks each independently into frame 19
- basis 17 and basis 18 install the exact same 23 landmark IDs:
  - intersection 23 / 23, Jaccard 1.000
  - 16 bootstrap and 7 map-growth landmarks in both
  - birth-frame split is identical: frame 1 = 16, frame 6 = 2, frame 9 = 5
  - frame 18 adds one observation to each shared landmark; it does not add or replace a landmark subset
- installed-set geometry does not show basis 18 becoming weaker:
  - centred 3D singular values are identical because the landmark set is identical: 17.262 / 3.787 / 2.484
  - depth median changes from 22.79 to 21.63 and coefficient of variation from 0.074 to 0.083
  - geometric pairwise camera-ray median increases from 6.89° to 7.30°
  - historical maximum viewpoint-angle median increases from 9.74° to 9.82°
  - maximum baseline/depth-ratio median increases from 0.623 to 0.669
- the same three near-collocated landmark pairs are present in both installed bases; local image-space collisions are not basis-18-specific
- frame-19 forward comparison also favours basis 18 slightly:
  - basis 17 yields 18 live correspondences; basis 18 yields 22, containing all 18 plus four
  - both fail PnP with zero accepted inliers
  - under a common locally propagated ETH3D reference pose, residual median / p90 is 6.07 / 14.65 px for basis 17 and 6.05 / 12.98 px for basis 18
  - support within 8 px is 12 / 18 versus 15 / 22
  - DLT smallest/largest singular ratio is 0.00598 versus 0.00623
  - pose-Jacobian condition is 603 versus 526
- the local reference is comparative rather than absolute: frame-17-to-18 validation differs by 2.08° rotation and 21.0° translation direction
- basis 18 is therefore not a worse depth set, a narrower viewpoint set, or a worse subset swap
- classification: basis 17 and basis 18 still lack a clear geometric separator

## Current next step
The shared-18 core is already fully fragile at 8 / 12 / 20 px. No single-point removal rescues it. The geometry drift is broad and systemic, driven by bad canonical rescue poses at frames 12 and 16. Audit the frame-16 rescue acceptance path to understand why the 20 px localisation-only rescue accepted a pose that is far worse than the local temporal interpolation on the same later-live landmarks.
