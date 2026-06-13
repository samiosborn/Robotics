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
- the failure is now classified as coherent 2D tracks with internally consistent assignments attached to a geometrically incompatible 3D support set
- frame-19 geometry drift is broad rather than concentrated in a landmark subgroup
- the common frame-level spikes point next to canonical rescue-pose drift, especially at frames 12 and 16

## Current open question
Why do the canonical rescue-frame poses at frames 12 and 16 become broadly incompatible with otherwise well-observed landmark histories?

## Best next step
Audit canonical rescue-pose drift at frames 12 and 16 against neighbouring accepted poses and the same 22 landmark histories, starting at frame 12.
