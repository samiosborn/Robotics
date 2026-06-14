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
