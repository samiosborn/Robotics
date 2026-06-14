# Project rules

## Tooling
- use `uv` only
- do not run raw `pip`
- do not modify dependency files unless explicitly asked
- prefer the smallest validation that proves the point
- keep the repo in a clean, trusted state

## Coding style
- use UK English spelling
- use `#` comments only
- do not use docstrings
- do not use inline end-of-line comments
- keep comments short and functional
- prefer explicit code over clever abstractions

## SLAM change discipline
- do one narrow production patch at a time
- do not patch multiple policy areas in one run
- do not change BA, rescue logic, promotion logic, and diagnostics all at once
- diagnosis-only runs must not change production behaviour
- structural cleanup runs must not change SLAM behaviour

## Dataset-boundary rules
- `src/datasets/image_sequence.py` is the neutral sequence contract
- `src/datasets/eth3d.py` remains ETH3D-specific
- keep ETH3D sequence loading, dataset defaults, output naming, and dataset-specific shell code in ETH3D-facing files
- only move code from `scripts/` into `src/` if it is:
  - reusable
  - pure or computation-heavy
  - stable enough to become shared infrastructure
- do not move unstable research probes, counterfactual helpers, or hard-coded frame experiments into `src/`

## Script versus src boundary
Keep in `scripts/`:
- `main()`
- CLI wiring
- output directory handling
- visualisation orchestration
- dataset-specific defaults
- experiment-specific probes
- counterfactual replay helpers
- frame-number-specific audits

Move to `src/` only when clearly justified:
- pure PnP diagnostic calculations
- pure support-funnel calculations
- pure reprojection summaries
- pure geometry-history summaries
- canonical state access helpers

## Experiment workflow
Every meaningful run should end with one label:
- kept
- reverted
- inconclusive

If kept:
- commit code
- update `exp/current_status.md` if trusted baseline changed
- append to `exp/experiment_log.md`

If reverted:
- restore the code
- append a short entry to `exp/experiment_log.md`

If inconclusive:
- usually restore the code
- log the result briefly

## Note handling
- `exp/current_status.md` is the human-facing trusted baseline summary
- `exp/experiment_log.md` is the detailed research log
- `.ai/AGENT_BRIEF.md` is the compact agent starting point
- `.ai/PROJECT_RULES.md` stores durable working rules

## Validation defaults
For structural or production changes, prefer this validation pattern unless the task requires more:
- `uv run python -m py_compile ...`
- `uv run python -m pytest tests/slam -q`
- `PYTHONPATH=. uv run python scripts/demo_frontend_eth3d.py`
- one focused ETH3D diagnostic run

## Current caution
Do not widen BA casually.
Current evidence points first to rescue-pose quality — specifically the frame-16 accepted rescue pose — not automatic need for stronger BA.
Do not suppress rescue refresh; it supports downstream support continuity.
