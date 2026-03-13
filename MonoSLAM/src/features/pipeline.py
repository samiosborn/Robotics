# src/features/pipeline.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from features.gradients import img_to_grey
from features.harris import harris_keypoints
from features.multiscale import build_multiscale_brief
from features.patches import extract_patches


# Per-frame extracted features
@dataclass(frozen=True)
class FrameFeatures:
    # Keypoints in image coordinates, shape (N,2)
    kps_xy: np.ndarray
    # Descriptor array
    # - NCC mode: (N,P,P) float
    # - BRIEF mode: (N,nbytes) uint8
    desc: np.ndarray
    # Optional pyramid level per keypoint
    level: np.ndarray | None = None
    # Optional orientation per keypoint (radians)
    angle: np.ndarray | None = None


def _feature_cfg(cfg: dict) -> dict:
    if isinstance(cfg, dict) and isinstance(cfg.get("features"), dict):
        return cfg["features"]
    return cfg


def _match_mode(cfg: dict) -> str:
    c = _feature_cfg(cfg)
    m = c.get("match")
    if isinstance(m, str):
        return m.lower()
    if isinstance(m, dict):
        mode = m.get("mode", m.get("type"))
        if isinstance(mode, str):
            return mode.lower()
    # Default to BRIEF for descriptor-led tracking
    return "brief"


def _img_cast_cfg(c: dict) -> tuple[float, type, dict, bool, bool, dict]:
    img_cfg = c["image"]
    eps = float(img_cfg.get("eps", 1e-8))
    dtype_name = img_cfg.get("dtype", "float64")
    dtype = getattr(np, dtype_name) if isinstance(dtype_name, str) else dtype_name
    luminance_weights = img_cfg["luminance_weights"]
    assume_srgb = bool(img_cfg.get("assume_srgb", True))
    normalise_01 = bool(img_cfg.get("normalise_01", True))
    eotf_params = img_cfg.get("eotf", {})
    return eps, dtype, luminance_weights, assume_srgb, normalise_01, eotf_params


# Detect and describe a single image using shared feature modules
def detect_and_describe_image(im: np.ndarray, cfg: dict) -> FrameFeatures:
    c = _feature_cfg(cfg)
    mode = _match_mode(c)

    # --- BRIEF ---
    if mode == "brief":
        brief_cfg = c.get("brief", {})
        ms_cfg = brief_cfg.get("multiscale", {})

        patch_size = int(brief_cfg.get("patch_size", brief_cfg.get("patch", 31)))
        brief_bits = int(brief_cfg.get("bits", brief_cfg.get("brief_bits", 256)))
        brief_seed = int(brief_cfg.get("seed", 0))
        brief_orient = bool(brief_cfg.get("orient", brief_cfg.get("brief_orient", False)))

        num_levels = int(ms_cfg.get("num_levels", brief_cfg.get("num_levels", 4)))
        scale_factor = float(ms_cfg.get("scale_factor", brief_cfg.get("scale_factor", 0.75)))
        min_size = int(ms_cfg.get("min_size", 32))
        max_kps_per_level = ms_cfg.get("max_kps_per_level", None)
        if max_kps_per_level is not None:
            max_kps_per_level = int(max_kps_per_level)

        ms = build_multiscale_brief(
            im,
            c,
            patch_size=patch_size,
            brief_bits=brief_bits,
            brief_seed=brief_seed,
            brief_orient=brief_orient,
            num_levels=num_levels,
            scale_factor=scale_factor,
            min_size=min_size,
            max_kps_per_level=max_kps_per_level,
        )

        kps_xy = np.asarray(ms.kps[:, :2], dtype=np.float64)
        desc = np.asarray(ms.desc, dtype=np.uint8)
        level = np.asarray(ms.level, dtype=np.int64)
        angle = None if ms.ori is None else np.asarray(ms.ori, dtype=np.float64)
        return FrameFeatures(kps_xy=kps_xy, desc=desc, level=level, angle=angle)

    # --- NCC ---
    if mode == "ncc":
        match_cfg = c.get("match", {})
        ncc_cfg = match_cfg.get("ncc", {}) if isinstance(match_cfg, dict) else {}

        patch_size = int(ncc_cfg.get("patch_size", ncc_cfg.get("patch", 21)))
        border_margin = int(ncc_cfg.get("border_margin", 0))

        eps, dtype, luminance_weights, assume_srgb, normalise_01, eotf_params = _img_cast_cfg(c)
        im_grey = img_to_grey(
            im,
            luminance_weights=luminance_weights,
            eotf_params=eotf_params,
            assume_srgb=assume_srgb,
            normalise_01=normalise_01,
            dtype=dtype,
            eps=eps,
        )
        h = harris_keypoints(im, c)
        pr = extract_patches(
            im_grey,
            np.asarray(h.kps),
            patch_size=patch_size,
            border_margin=border_margin,
            normalise=True,
            dtype=dtype,
            eps=eps,
        )

        kps_xy = np.asarray(pr.kps[:, :2], dtype=np.float64) if pr.kps.size else np.zeros((0, 2), dtype=np.float64)
        desc = np.asarray(pr.patches, dtype=np.float64)
        return FrameFeatures(kps_xy=kps_xy, desc=desc, level=None, angle=None)

    raise ValueError(f"Unknown match mode '{mode}'; expected 'ncc' or 'brief'")
