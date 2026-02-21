# src/features/matching.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from features.checks import check_3d_patches, check_finite_scalar, check_int_gt0


# Matching results return
@dataclass(frozen=True)
class MatchResult:
    # Indices into set A (0, ..., Na-1), shape (M,)
    ia: np.ndarray
    # Indices into set B (0, ..., Nb-1), shape (M,)
    ib: np.ndarray
    # Match scores, shape (M,)
    score: np.ndarray


# Flatten patches (N, P, P) -> (N, D)
def flatten_patches(patches, *, dtype=np.float64):

    # Checks
    patches = check_3d_patches(patches, name="patches", finite=True)

    # Read dimensions
    N, P, _ = patches.shape

    # Cast
    X = np.asarray(patches, dtype=dtype)

    # Reshape into vectors
    X = X.reshape(N, P * P)

    return X


# L2 normalise descriptors (row-wise)
def l2_normalise_rows(X, *, eps=1e-8):

    # --- Checks ---
    # Require numpy array
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    # Require 2D
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}")
    # Enforce finite values
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    # Validate eps
    eps = check_finite_scalar(eps, "eps")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps}")

    # Compute row norms
    n = np.linalg.norm(X, axis=1)

    # Normalise rows
    n = np.maximum(n, eps)
    Xn = X / n[:, None]

    return Xn


# Z-score normalise rows (mean 0, std 1)
def zscore_rows(X, *, eps=1e-8):

    # --- Checks ---
    # Require numpy array
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    # Require 2D
    if X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    # Enforce finite values
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    # Validate eps
    eps = check_finite_scalar(eps, "eps")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps}")

    # Compute per-row mean
    mu = X.mean(axis=1, keepdims=True)

    # Subtract mean
    Z = X - mu

    # Compute per-row std
    sigma = Z.std(axis=1, keepdims=True)

    # Divide by std
    sigma = np.maximum(sigma, eps)
    Z = Z / sigma

    return Z


# Compute NCC similarity matrix
def ncc_matrix(
    patchesA,
    patchesB,
    *,
    eps=1e-8,
    dtype=np.float64,
    assume_normalised=False,
):

    # --- Checks ---
    # Validate eps
    eps = check_finite_scalar(eps, "eps")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps}")
    # Validate patches tensors (N,P,P)
    patchesA = check_3d_patches(patchesA, name="patchesA", finite=True)
    patchesB = check_3d_patches(patchesB, name="patchesB", finite=True)
    # Require same patch size
    if patchesA.shape[1:] != patchesB.shape[1:]:
        raise ValueError(f"patch sizes must match; got {patchesA.shape[1:]} and {patchesB.shape[1:]}")

    # Flatten patches into vectors
    XA = flatten_patches(patchesA, dtype=dtype)
    XB = flatten_patches(patchesB, dtype=dtype)

    # If patches are already normalised, skip z-scoring
    if bool(assume_normalised):
        ZA = XA
        ZB = XB
    else:
        # Z-score each patch vector (mean 0, std 1)
        ZA = zscore_rows(XA, eps=eps)
        ZB = zscore_rows(XB, eps=eps)

    # L2 normalise so dot product
    ZA = l2_normalise_rows(ZA, eps=eps)
    ZB = l2_normalise_rows(ZB, eps=eps)

    # Similarity matrix S = ZA @ ZB^T
    S = np.clip(ZA @ ZB.T, -1.0, 1.0)

    return S


# Match patches using NCC
def match_patches_ncc(
    patchesA,
    patchesB,
    *,
    min_score=0.7,
    mutual=True,
    max_matches=None,
    eps=1e-8,
    dtype=np.float64,
    assume_normalised=False,
) -> MatchResult:

    # --- Checks ---
    # Validate min_score
    min_score = check_finite_scalar(min_score, "min_score")
    # Keep min_score in [-1, 1]
    if min_score < -1.0 or min_score > 1.0:
        raise ValueError(f"min_score should be in [-1,1] for NCC; got {min_score}")
    # Validate mutual flag
    mutual = bool(mutual)
    # Validate max_matches if provided
    if max_matches is not None:
        max_matches = check_int_gt0(max_matches, "max_matches")

    # Compute similarity matrix (Na, Nb)
    S = ncc_matrix(
        patchesA,
        patchesB,
        eps=eps,
        dtype=dtype,
        assume_normalised=assume_normalised)

    # Read sizes
    Na, Nb = S.shape

    # Early exit if either side is empty
    if Na == 0 or Nb == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64))

    # For each A, find best B
    ib_best = np.argmax(S, axis=1)

    # Best scores for each A
    s_best = S[np.arange(Na), ib_best]

    # Threshold by minimum acceptable similarity
    keep = (s_best >= float(min_score))

    # Candidate matches
    ia = np.nonzero(keep)[0].astype(np.int64, copy=False)
    ib = ib_best[keep].astype(np.int64, copy=False)
    score = s_best[keep].astype(np.float64, copy=False)

    # If mutual matching requested, enforce B also picks A
    if mutual and ia.size > 0:
        # For each B, find best A
        ia_best_for_b = np.argmax(S, axis=0)
        # Keep only pairs where A is also best for that B
        ok = (ia == ia_best_for_b[ib])
        ia = ia[ok]
        ib = ib[ok]
        score = score[ok]

    # If nothing remains, return empty
    if score.size == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Sort matches by score descending
    order = np.argsort(score)[::-1]
    ia = ia[order]
    ib = ib[order]
    score = score[order]

    # Truncate to max_matches
    if max_matches is not None:
        ia = ia[:max_matches]
        ib = ib[:max_matches]
        score = score[:max_matches]

    return MatchResult(ia=ia, ib=ib, score=score)


# Compute L2 distance matrix between float descriptors
def l2_distance_matrix(descA, descB, *, dtype=np.float64):

    # --- Checks ---
    # Require numpy arrays
    if not isinstance(descA, np.ndarray) or not isinstance(descB, np.ndarray):
        raise ValueError("descA and descB must be numpy arrays")
    # Require 2D
    if descA.ndim != 2 or descB.ndim != 2:
        raise ValueError(f"descA and descB must be 2D; got {descA.shape} and {descB.shape}")
    # Require same descriptor dimension
    if descA.shape[1] != descB.shape[1]:
        raise ValueError(f"Descriptor dims must match; got {descA.shape[1]} and {descB.shape[1]}")
    # Cast
    A = np.asarray(descA, dtype=dtype)
    B = np.asarray(descB, dtype=dtype)
    # Enforce finite values
    if not np.isfinite(A).all():
        raise ValueError("descA must contain only finite values")
    if not np.isfinite(B).all():
        raise ValueError("descB must contain only finite values")

    # Early exit for empties
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)

    # Compute squared norms
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T

    # Compute pairwise squared distances
    D2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)

    return np.sqrt(D2)


# Match float descriptors using Lowe-style ratio test (SIFT-style)
def match_descriptors_l2_ratio(
    descA,
    descB,
    *,
    ratio=0.8,
    mutual=True,
    max_matches=None,
    dtype=np.float64,
) -> MatchResult:

    # --- Checks ---
    # Validate ratio
    ratio = check_finite_scalar(ratio, "ratio")
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"ratio must be in (0,1); got {ratio}")
    # Validate mutual flag
    mutual = bool(mutual)
    # Validate max_matches if provided
    if max_matches is not None:
        max_matches = check_int_gt0(max_matches, "max_matches")

    # Compute distance matrix (Na, Nb)
    D = l2_distance_matrix(descA, descB, dtype=dtype)

    # Read sizes
    Na, Nb = D.shape

    # Early exit
    if Na == 0 or Nb == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64))

    # If only one candidate per row exists, ratio test is not defined
    if Nb < 2:
        # Best match for each A
        j1 = np.argmin(D, axis=1)
        # Distances of best match
        d1 = D[np.arange(Na), j1]
        # Keep all (no ratio possible)
        ia = np.arange(Na, dtype=np.int64)
        ib = j1.astype(np.int64, copy=False)
        # Score = negative distance (higher is better)
        score = (-d1).astype(np.float64, copy=False)

    else:
        # Get indices of the two nearest neighbours in B for each A (fast top-2)
        idx2 = np.argpartition(D, kth=1, axis=1)[:, :2]

        # Gather the two candidate distances
        d2cand = D[np.arange(Na)[:, None], idx2]

        # Sort the two candidates so column 0 is nearest, column 1 is second-nearest
        order2 = np.argsort(d2cand, axis=1)
        idx_sorted = idx2[np.arange(Na)[:, None], order2]
        d_sorted = d2cand[np.arange(Na)[:, None], order2]

        # Nearest neighbour index and distance
        j1 = idx_sorted[:, 0]
        d1 = d_sorted[:, 0]

        # Second-nearest distance
        d2 = np.maximum(d_sorted[:, 1], 1e-12)

        # Apply ratio test: d1 / d2 < ratio
        keep = (d1 / d2) < float(ratio)

        # Keep candidates
        ia = np.nonzero(keep)[0].astype(np.int64, copy=False)
        ib = j1[keep].astype(np.int64, copy=False)

        # Score = negative distance so higher is better
        score = (-d1[keep]).astype(np.float64, copy=False)

    # Enforce mutual best if requested
    if mutual and score.size > 0:
        # Best A for each B is argmin over A
        ia_best_for_b = np.argmin(D, axis=0)
        # Keep only pairs where A is also best for that B
        ok = (ia == ia_best_for_b[ib])
        ia = ia[ok]
        ib = ib[ok]
        score = score[ok]

    # If nothing remains
    if score.size == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Sort by score descending
    order = np.argsort(score)[::-1]
    ia = ia[order]
    ib = ib[order]
    score = score[order]

    # Truncate
    if max_matches is not None:
        ia = ia[:max_matches]
        ib = ib[:max_matches]
        score = score[:max_matches]

    return MatchResult(ia=ia, ib=ib, score=score)
