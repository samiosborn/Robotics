# scripts/frontend_eth3d_common.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.checks import check_file
from slam.pnp_config import pnp_frontend_kwargs_from_cfg, pnp_threshold_stability_cfg_from_pnp, pnp_threshold_stability_defaults
from utils.load_config import load_config

STANDARD_FRAME_STAT_FIELDS = (
    "frame_index",
    "reference_keyframe_index",
    "pipeline_ok",
    "pipeline_reason",
    "n_track_inliers",
    "n_pnp_corr",
    "n_pnp_inliers",
    "n_new_added",
    "pipeline_keyframe_promoted",
    "seed_landmarks_before",
    "seed_landmarks_after",
    "rescue_attempted",
    "rescue_succeeded",
    "localisation_only_rescue_frame",
    "support_refresh_triggered",
)


# Load greyscale image for visualisation
def load_pil_greyscale(path: Path) -> Image.Image:
    return Image.open(str(check_file(path, name="image"))).convert("L")


# Resolve profile include path
def resolve_include_path(rel_path: str, profile_path: Path) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return check_file(p, name="include")

    # Profile includes are written from repo root
    return check_file(ROOT / p, name="include")


# Add threshold-stability command-line overrides
def add_pnp_threshold_stability_args(parser: argparse.ArgumentParser) -> None:
    # Enable the diagnostic threshold-stability comparison
    parser.add_argument("--enable_pnp_threshold_stability_diagnostic", action=argparse.BooleanOptionalAction, default=None)

    # Apply the threshold-stability diagnostic as an optional frontend guard
    parser.add_argument("--enable_pnp_threshold_stability_gate", action=argparse.BooleanOptionalAction, default=None)

    # Comparison threshold for the accepted PnP pose
    parser.add_argument("--pnp_threshold_stability_compare_px", type=float, default=None)

    # Minimum support IoU before the accepted pose is flagged unstable
    parser.add_argument("--pnp_threshold_stability_min_support_iou", type=float, default=None)

    # Maximum translation-direction disagreement before the accepted pose is flagged unstable
    parser.add_argument("--pnp_threshold_stability_max_translation_direction_deg", type=float, default=None)

    # Maximum camera-centre direction disagreement before the accepted pose is flagged unstable
    parser.add_argument(
        "--pnp_threshold_stability_max_camera_centre_direction_deg",
        dest="pnp_threshold_stability_max_camera_centre_direction_deg",
        type=float,
        default=None,
    )

    # Support IoU at or below which threshold supports are treated as disjoint
    parser.add_argument("--pnp_threshold_stability_disjoint_iou", type=float, default=None)


# Apply explicit threshold-stability command-line overrides
def apply_pnp_threshold_stability_cli_overrides(pnp_cfg: dict, args: argparse.Namespace) -> dict:
    out = dict(pnp_cfg)
    for key in pnp_threshold_stability_defaults().keys():
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value

    out.update(pnp_threshold_stability_cfg_from_pnp(out))

    return out


# Load profile and build runtime config
def load_runtime_cfg(profile_path: Path) -> tuple[dict, np.ndarray]:
    profile_cfg = load_config(check_file(profile_path, name="profile"))

    includes = profile_cfg["includes"]
    camera_cfg = load_config(resolve_include_path(includes["camera"], profile_path))
    features_cfg = load_config(resolve_include_path(includes["features"], profile_path))
    bootstrap_cfg = load_config(resolve_include_path(includes["bootstrap"], profile_path))

    K_cfg = camera_cfg["camera"]["K"]
    K = np.array(
        [
            [float(K_cfg["fx"]), 0.0, float(K_cfg["cx"])],
            [0.0, float(K_cfg["fy"]), float(K_cfg["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    runtime_cfg = {
        "profile": profile_cfg.get("profile", {}),
        "dataset": profile_cfg.get("dataset", {}),
        "run": profile_cfg.get("run", {}),
        "features": features_cfg,
        "ransac": bootstrap_cfg["ransac"],
        "bootstrap": bootstrap_cfg["bootstrap"],
        "pnp": bootstrap_cfg.get("pnp", {}),
    }

    return runtime_cfg, K


# Build keyword arguments for the current frontend and tracking APIs
def frontend_kwargs_from_cfg(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dict")

    features_cfg = cfg["features"]
    ransac_cfg = cfg["ransac"]
    bootstrap_cfg = cfg["bootstrap"]
    pnp_cfg = cfg.get("pnp", {})
    pnp_frontend_kwargs = pnp_frontend_kwargs_from_cfg(pnp_cfg)

    return {
        "feature_cfg": features_cfg,
        "F_cfg": ransac_cfg["F"],
        "H_cfg": ransac_cfg["H"],
        "bootstrap_cfg": bootstrap_cfg,
        "pnp_threshold_px": float(pnp_frontend_kwargs["threshold_px"]),
        "pnp_frontend_kwargs": pnp_frontend_kwargs,
    }


# Count landmarks in a seed dictionary
def seed_landmark_count(seed: dict | None) -> int:
    if not isinstance(seed, dict):
        return 0

    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        return 0

    return int(len(landmarks))


# Build the standard per-frame diagnostic fields
def standard_frame_stats(
    *,
    frame_index: int,
    reference_keyframe_index: int | None,
    frontend_out: dict | None = None,
    stats: dict | None = None,
    seed_before: dict | None = None,
    seed_after: dict | None = None,
    seed_landmarks_before: int | None = None,
    seed_landmarks_after: int | None = None,
) -> dict:
    frontend_out = frontend_out if isinstance(frontend_out, dict) else {}
    stats = stats if isinstance(stats, dict) else frontend_out.get("stats", {})
    stats = stats if isinstance(stats, dict) else {}

    if seed_landmarks_before is None:
        seed_landmarks_before = seed_landmark_count(seed_before)

    if seed_landmarks_after is None:
        if "seed_landmarks_after" in stats:
            seed_landmarks_after = int(stats.get("seed_landmarks_after", 0))
        else:
            seed_landmarks_after = seed_landmark_count(seed_after if seed_after is not None else frontend_out.get("seed", {}))

    return {
        "frame_index": int(frame_index),
        "reference_keyframe_index": None if reference_keyframe_index is None else int(reference_keyframe_index),
        "pipeline_ok": bool(frontend_out.get("ok", stats.get("ok", False))),
        "pipeline_reason": stats.get("reason", None),
        "n_track_inliers": int(stats.get("n_track_inliers", 0)),
        "n_pnp_corr": int(stats.get("n_pnp_corr", 0)),
        "n_pnp_inliers": int(stats.get("n_pnp_inliers", 0)),
        "n_new_added": int(stats.get("n_new_added", 0)),
        "pipeline_keyframe_promoted": bool(stats.get("keyframe_promoted", stats.get("pipeline_keyframe_promoted", False))),
        "seed_landmarks_before": int(seed_landmarks_before),
        "seed_landmarks_after": int(seed_landmarks_after),
        "rescue_attempted": bool(stats.get("pnp_support_rescue_attempted", stats.get("rescue_attempted", False))),
        "rescue_succeeded": bool(stats.get("pnp_support_rescue_succeeded", stats.get("rescue_succeeded", False))),
        "localisation_only_rescue_frame": bool(stats.get("localisation_only_rescue_frame", False)),
        "support_refresh_triggered": bool(stats.get("guarded_support_refresh_triggered", stats.get("support_refresh_triggered", False))),
    }


# Keep one parseable scorecard subset stable
def frame_scorecard_row(row: dict) -> dict:
    return {key: row.get(key, None) for key in STANDARD_FRAME_STAT_FIELDS}


# Format the standard per-frame console line
def format_frame_scorecard(row: dict, *, mode: str = "short") -> str:
    ref = row.get("reference_keyframe_index", None)
    ref_text = "None" if ref is None else str(int(ref))
    reason = row.get("pipeline_reason", None)
    line = (
        f"frame {int(row.get('frame_index', -1)):04d} "
        f"ref={ref_text} "
        f"ok={bool(row.get('pipeline_ok', False))} "
        f"reason={reason} "
        f"track={int(row.get('n_track_inliers', 0))} "
        f"pnp={int(row.get('n_pnp_inliers', 0))}/{int(row.get('n_pnp_corr', 0))} "
        f"new={int(row.get('n_new_added', 0))} "
        f"kf={bool(row.get('pipeline_keyframe_promoted', False))} "
        f"rescue={bool(row.get('rescue_attempted', False))}/{bool(row.get('rescue_succeeded', False))} "
        f"refresh={bool(row.get('support_refresh_triggered', False))} "
        f"landmarks={int(row.get('seed_landmarks_before', 0))}->{int(row.get('seed_landmarks_after', 0))}"
    )

    if str(mode) != "long":
        return line

    extras = []
    for key in [
        "diagnostic_n_pnp_corr",
        "diagnostic_n_pnp_inliers",
        "pnp_spatial_gate_rejected",
        "pnp_component_gate_rejected",
        "threshold_summary",
    ]:
        if key in row:
            extras.append(f"{key}={row[key]}")

    if len(extras) == 0:
        return line

    return f"{line} " + " ".join(extras)


# Start a fresh JSONL file for one diagnostic run
def reset_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


# Append one JSONL record
def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
