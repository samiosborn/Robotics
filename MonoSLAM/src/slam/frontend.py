# src/slam/frontend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Per-frame extracted features
@dataclass(frozen=True)
class FrameFeatures:
    # Conventions:
    # - kps_xy: (N,2) float, columns [x,y]
    # - desc: descriptor array (binary packed uint8 for BRIEF, or float for NCC patches)
    # - level: optional pyramid level per keypoint (N,) int, if multiscale is used
    # - angle: optional orientation per keypoint (N,) float radians, if orientation is used
    kps_xy: np.ndarray
    desc: np.ndarray
    level: np.ndarray | None = None
    angle: np.ndarray | None = None


# Output of matching two frames or two sets of features
@dataclass(frozen=True)
class MatchBundle:
    # Conventions:
    # - ia, ib index into A and B respectively
    # - score higher is better (NCC similarity or negative distance)
    ia: np.ndarray
    ib: np.ndarray
    score: np.ndarray


# Output of a two-view estimation step
@dataclass(frozen=True)
class TwoViewEstimation:
    # Typical usage:
    # - Estimate F with RANSAC
    # - Optionally estimate H and check planar degeneracy
    # - Recover pose (R,t) via E decomposition + cheirality check
    F: np.ndarray | None
    H: np.ndarray | None
    inlier_mask: np.ndarray | None
    R: np.ndarray | None
    t: np.ndarray | None
    E: np.ndarray | None
    cheirality_mask: np.ndarray | None
    stats: dict[str, Any]



# Produce keypoints + descriptors for a single greyscale image
def detect_and_describe(im: np.ndarray, cfg: dict[str, Any]) -> FrameFeatures:
    # Intended behaviour:
    # - Run detector (Harris or Shi-Tomasi) using cfg["harris"] etc.
    # - Optionally build pyramid and run per-level detection if cfg enables multiscale
    # - Optionally compute orientation if cfg enables it
    # - Compute descriptors:
    #   - BRIEF packed bits (uint8) if match mode is brief
    #   - Or patches (float) if match mode is ncc
    # Returns:
    # - FrameFeatures(kps_xy, desc, level, angle)
    raise NotImplementedError


# Match two feature sets given a chosen matcher mode
def match_frames(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    cfg: dict[str, Any],
) -> MatchBundle:
    # Intended behaviour:
    # - cfg controls:
    #   - match mode: "ncc" or "brief"
    #   - thresholds: min_score, max_dist, ratio, mutual, max_matches
    #   - multiscale gating via "scale_gate" if BRIEF multiscale is enabled
    # Returns:
    # - MatchBundle(ia, ib, score)
    raise NotImplementedError


# Return matched (x,y) arrays
def matched_keypoints_xy(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    matches: MatchBundle,
) -> tuple[np.ndarray, np.ndarray]:
    # Returns:
    # - xyA: (M,2)
    # - xyB: (M,2)
    raise NotImplementedError


# Two-view estimation from matched keypoints
def estimate_two_view(
    K1: np.ndarray,
    K2: np.ndarray,
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    matches: MatchBundle,
    cfg: dict[str, Any],
) -> TwoViewEstimation:
    # Intended behaviour:
    # - Convert matched xy to (2,N) point matrices if needed
    # - Estimate F with RANSAC using cfg["ransac"]["F"]
    # - Optionally estimate H with RANSAC using cfg["ransac"]["H"]
    # - Planar degeneracy check (H inliers dominate F inliers)
    # - Recover pose (R,t,E,cheirality_mask) from F + intrinsics
    # Returns:
    # - TwoViewEstimation(...)
    raise NotImplementedError


# Bootstrap from two-frames
def bootstrap_from_two_frames(
    K1: np.ndarray,
    K2: np.ndarray,
    im0: np.ndarray,
    im1: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    # Intended behaviour:
    # - detect_and_describe on im0 and im1
    # - match_frames
    # - estimate_two_view
    # - call slam.bootstrap.bootstrap_two_view to triangulate and initialise map seed
    # Returns:
    # - seed dict (same shape/convention as slam/bootstrap.py)
    raise NotImplementedError


# Track against keyframe
def track_against_keyframe(
    K: np.ndarray,
    keyframe_feats: FrameFeatures,
    cur_im: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    # Intended behaviour:
    # - detect_and_describe on cur_im
    # - match keyframe -> current
    # - run a geometric check (F RANSAC) to get inliers
    # - return tracks that can be used for:
    #   - pose-only optimisation (PnP later when 2D-3D is available)
    #   - keyframe decision logic
    # Returns:
    # - dict with:
    #   - cur_feats
    #   - matches
    #   - inlier_mask
    #   - xy_kf, xy_cur (matched/inlier)
    #   - stats
    raise NotImplementedError

