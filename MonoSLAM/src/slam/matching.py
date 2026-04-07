# src/slam/matching.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import check_choice, check_finite_scalar, check_in_01, check_int_ge0, check_int_gt0, check_points_xy_N2plus
from features.matching import match_brief_hamming_with_scale_gate, match_patches_ncc
from features.pipeline import FrameFeatures


# Matching result
@dataclass(frozen=True)
class MatchBundle:
    ia: np.ndarray
    ib: np.ndarray
    score: np.ndarray


# Match two feature bundles while preserving original feature indices
def match_frames(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    *,
    mode: str | None = None,
    ncc_min_score: float = 0.7,
    brief_mode: str = "nn",
    brief_max_dist: int | None = 80,
    brief_ratio: float = 0.8,
    mutual: bool = True,
    max_matches: int | None = None,
    scale_gate: int = 1,
) -> MatchBundle:
    # Read keypoints and descriptors
    kpsA = np.asarray(featsA.kps_xy)
    kpsB = np.asarray(featsB.kps_xy)
    descA = np.asarray(featsA.desc)
    descB = np.asarray(featsB.desc)

    # Early exit on empty features
    if kpsA.shape[0] == 0 or kpsB.shape[0] == 0:
        return MatchBundle(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )
    if descA.shape[0] == 0 or descB.shape[0] == 0:
        return MatchBundle(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Infer the mode from descriptor structure when it is not supplied
    if mode is None:
        if descA.ndim == 3 and descB.ndim == 3:
            mode = "ncc"
        else:
            mode = "brief"

    # Validate common controls
    mode = check_choice(mode, {"ncc", "brief"}, name="match mode")
    mutual = bool(mutual)

    if max_matches is not None:
        max_matches = check_int_gt0(max_matches, name="max_matches")

    # NCC match mode
    if mode == "ncc":
        # Validate NCC controls
        ncc_min_score = check_finite_scalar(ncc_min_score, name="ncc_min_score")
        check_in_01(ncc_min_score, name="ncc_min_score", eps=0.0)

        # Run patch NCC matching
        m = match_patches_ncc(
            descA,
            descB,
            min_score=ncc_min_score,
            mutual=mutual,
            max_matches=max_matches,
        )

        return MatchBundle(
            ia=np.asarray(m.ia, dtype=np.int64),
            ib=np.asarray(m.ib, dtype=np.int64),
            score=np.asarray(m.score, dtype=np.float64),
        )

    # Validate BRIEF controls
    brief_mode = check_choice(brief_mode, {"nn", "ratio"}, name="brief mode")

    if brief_max_dist is not None:
        brief_max_dist = check_int_ge0(brief_max_dist, name="brief_max_dist")

    brief_ratio = check_finite_scalar(brief_ratio, name="brief_ratio")
    if brief_ratio <= 0.0 or brief_ratio >= 1.0:
        raise ValueError(f"brief_ratio must be in (0,1); got {brief_ratio}")

    scale_gate = check_int_ge0(scale_gate, name="scale_gate")

    # Read pyramid levels with a safe zero default
    lvlA = np.asarray(featsA.level, dtype=np.int64).reshape(-1) if featsA.level is not None else np.zeros((descA.shape[0],), dtype=np.int64)
    lvlB = np.asarray(featsB.level, dtype=np.int64).reshape(-1) if featsB.level is not None else np.zeros((descB.shape[0],), dtype=np.int64)

    # Run BRIEF Hamming matching with a scale gate
    m = match_brief_hamming_with_scale_gate(
        descA,
        lvlA,
        descB,
        lvlB,
        mode=brief_mode,
        max_dist=brief_max_dist,
        ratio=brief_ratio,
        mutual=mutual,
        max_matches=max_matches,
        scale_gate=scale_gate,
    )

    return MatchBundle(
        ia=np.asarray(m.ia, dtype=np.int64),
        ib=np.asarray(m.ib, dtype=np.int64),
        score=np.asarray(m.score, dtype=np.float64),
    )


# Convert matched feature indices into matched coordinate pairs
def matched_keypoints_xy(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    matches: MatchBundle,
) -> tuple[np.ndarray, np.ndarray]:
    # Check match index arrays
    ia = np.asarray(matches.ia, dtype=np.int64).reshape(-1)
    ib = np.asarray(matches.ib, dtype=np.int64).reshape(-1)

    # Require matched index arrays to agree
    if ia.size != ib.size:
        raise ValueError(f"matches.ia and matches.ib must have equal size; got {ia.size} and {ib.size}")

    # Early exit on empty matches
    if ia.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    # Check keypoint arrays
    kpsA = check_points_xy_N2plus(featsA.kps_xy, name="featsA.kps_xy", dtype=float, finite=True)
    kpsB = check_points_xy_N2plus(featsB.kps_xy, name="featsB.kps_xy", dtype=float, finite=True)

    # Check feature index bounds
    if int(ia.min()) < 0 or int(ia.max()) >= int(kpsA.shape[0]):
        raise ValueError("matches.ia contains out-of-range feature indices")
    if int(ib.min()) < 0 or int(ib.max()) >= int(kpsB.shape[0]):
        raise ValueError("matches.ib contains out-of-range feature indices")

    # Gather matched image coordinates
    xyA = np.asarray(kpsA[ia, :2], dtype=np.float64)
    xyB = np.asarray(kpsB[ib, :2], dtype=np.float64)

    return xyA, xyB
