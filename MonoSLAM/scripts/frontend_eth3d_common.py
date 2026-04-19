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


# Append one JSONL record
def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
