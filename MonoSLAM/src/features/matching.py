# src/features/matching.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from features.checks import check_finite_scalar, check_int_gt0


# Return bundle for matching results
@dataclass(frozen=True)
class MatchResult:
    # Indices into set A (0..Na-1), shape (M,)
    ia: np.ndarray
    # Indices into set B (0..Nb-1), shape (M,)
    ib: np.ndarray
    # Match scores (higher is better), shape (M,)
    score: np.ndarray


# Flatten patches (N, P, P) -> (N, D)
def flatten_patches(patches, *, dtype=np.float64):

    # --- Checks ---
    # Require numpy array
    if not isinstance(patches, np.ndarray):
        raise ValueError("patches must be a numpy array")
    # Require 3D tensor
    if patches.ndim != 3:
        raise ValueError(f"patches must have shape (N,P,P); got {patches.shape}")
    # Read dimensions
    N, P1, P2 = patches.shape
    # Require square patches
    if P1 != P2:
        raise ValueError(f"patches must be square; got {P1}x{P2}")

    # Cast to float dtype for dot products
    X = np.asarray(patches, dtype=dtype)

    # Reshape into vectors
    X = X.reshape(N, P1 * P2)

    return X


# L2 normalise descriptors row-wise (used for cosine / NCC-style dot product)
def l2_normalise_rows(X, *, eps=1e-8):

    # --- Checks ---
    # Require numpy array
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    # Require 2D
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}")
    # Validate eps
    eps = float(eps)
    if (not np.isfinite(eps)) or eps <= 0.0:
        raise ValueError(f"eps must be finite and > 0; got {eps}")

    # Compute row norms
    n = np.linalg.norm(X, axis=1)

    # Avoid divide-by-zero
    n = np.maximum(n, eps)

    # Normalise rows
    Xn = X / n[:, None]

    return Xn


# Ensure patches are normalised for NCC (zero-mean, unit-std per patch)
def zscore_rows(X, *, eps=1e-8):

    # --- Checks ---
    # Require 2D
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    # Validate eps
    eps = float(eps)
    if (not np.isfinite(eps)) or eps <= 0.0:
        raise ValueError(f"eps must be finite and > 0; got {eps}")

    # Compute per-row mean
    mu = X.mean(axis=1, keepdims=True)

    # Subtract mean
    Z = X - mu

    # Compute per-row std
    sigma = Z.std(axis=1, keepdims=True)

    # Stabilise std
    sigma = np.maximum(sigma, eps)

    # Divide by std
    Z = Z / sigma

    return Z


# Compute full NCC similarity matrix between A and B
def ncc_matrix(patchesA, patchesB, *, eps=1e-8, dtype=np.float64, assume_normalised=False):

    # Flatten patches into vectors
    XA = flatten_patches(patchesA, dtype=dtype)
    XB = flatten_patches(patchesB, dtype=dtype)

    # Z-score each patch vector (mean 0, std 1) unless already normalised
    if bool(assume_normalised):
        ZA = XA
        ZB = XB
    else:
        ZA = zscore_rows(XA, eps=eps)
        ZB = zscore_rows(XB, eps=eps)

    # L2 normalise after z-score so dot product equals cosine/NCC
    ZA = l2_normalise_rows(ZA, eps=eps)
    ZB = l2_normalise_rows(ZB, eps=eps)

    # Similarity matrix S = ZA @ ZB^T
    S = ZA @ ZB.T

    # Clamp small numerical spill
    S = np.clip(S, -1.0, 1.0)

    return S


# Match patches using NCC (higher is better)
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
    # Validate mutual flag
    mutual = bool(mutual)
    # Validate max_matches if provided
    if max_matches is not None:
        max_matches = check_int_gt0(max_matches, "max_matches")

    # Compute similarity matrix (Na, Nb)
    S = ncc_matrix(patchesA, patchesB, eps=eps, dtype=dtype, assume_normalised=assume_normalised)

    # Read sizes
    Na, Nb = S.shape

    # Early exit if either side is empty
    if Na == 0 or Nb == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

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

    # Truncate to max_matches if requested
    if max_matches is not None:
        ia = ia[:max_matches]
        ib = ib[:max_matches]
        score = score[:max_matches]

    return MatchResult(ia=ia, ib=ib, score=score)


# Compute L2 distance matrix between float descriptors (lower is better)
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

    # Early exit for empties
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)

    # Compute squared norms
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T

    # Compute pairwise squared distances using ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    D2 = a2 + b2 - 2.0 * (A @ B.T)

    # Clamp numerical negatives
    D2 = np.maximum(D2, 0.0)

    # Return L2 distances
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
            score=np.zeros((0,), dtype=np.float64),
        )

    # If only one candidate per row exists, ratio test is not defined
    if Nb < 2:
        # Best match for each A
        j1 = np.argmin(D, axis=1)
        # Distances of best match
        d1 = D[np.arange(Na), j1]
        # Keep all (no ratio possible)
        ia = np.arange(Na, dtype=np.int64)
        ib = j1.astype(np.int64, copy=False)
        # Score = negative distance so higher is better
        score = (-d1).astype(np.float64, copy=False)

    else:
        # Get indices of the two nearest neighbours in B for each A
        idx2 = np.argpartition(D, kth=1, axis=1)[:, :2]

        # Gather the two candidate distances
        d2cand = D[np.arange(Na)[:, None], idx2]

        # Sort the two candidates so col 0 is nearest, col 1 is second-nearest
        order2 = np.argsort(d2cand, axis=1)
        idx_sorted = idx2[np.arange(Na)[:, None], order2]
        d_sorted = d2cand[np.arange(Na)[:, None], order2]

        # Nearest neighbour index and distance
        j1 = idx_sorted[:, 0]
        d1 = d_sorted[:, 0]

        # Second-nearest distance
        d2 = d_sorted[:, 1]

        # Stabilise denominator
        d2 = np.maximum(d2, 1e-12)

        # Apply ratio test
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

    # Sort by score descending (i.e., smallest distance first)
    order = np.argsort(score)[::-1]
    ia = ia[order]
    ib = ib[order]
    score = score[order]

    # Truncate if requested
    if max_matches is not None:
        ia = ia[:max_matches]
        ib = ib[:max_matches]
        score = score[:max_matches]

    return MatchResult(ia=ia, ib=ib, score=score)


# Precompute a popcount lookup table for uint8 values 0..255
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


# Compute Hamming distance matrix between packed uint8 descriptors
def hamming_distance_matrix(descA: np.ndarray, descB: np.ndarray) -> np.ndarray:

    # --- Checks ---
    # Require numpy arrays
    if not isinstance(descA, np.ndarray) or not isinstance(descB, np.ndarray):
        raise ValueError("descA and descB must be numpy arrays")
    # Require 2D
    if descA.ndim != 2 or descB.ndim != 2:
        raise ValueError(f"descA and descB must be 2D; got {descA.shape} and {descB.shape}")
    # Require uint8 packed descriptors
    if descA.dtype != np.uint8 or descB.dtype != np.uint8:
        raise ValueError("descA and descB must be uint8 packed descriptors (use np.packbits)")
    # Require same descriptor byte length
    if descA.shape[1] != descB.shape[1]:
        raise ValueError(f"descriptor byte dims must match; got {descA.shape[1]} and {descB.shape[1]}")

    # Read sizes
    Na = int(descA.shape[0])
    Nb = int(descB.shape[0])

    # Early exit
    if Na == 0 or Nb == 0:
        return np.zeros((Na, Nb), dtype=np.int32)

    # XOR broadcasts to (Na, Nb, nbytes)
    x = descA[:, None, :] ^ descB[None, :, :]

    # Lookup popcount per byte, sum over bytes -> (Na, Nb)
    D = _POPCOUNT_LUT[x].sum(axis=2, dtype=np.int32)

    return D


# Match packed binary descriptors using nearest-neighbour + distance threshold
def match_descriptors_hamming_nn(
    descA: np.ndarray,
    descB: np.ndarray,
    *,
    max_dist: int | None = 80,
    mutual: bool = True,
    max_matches: int | None = None,
) -> MatchResult:

    # --- Checks ---
    # Validate mutual flag
    mutual = bool(mutual)
    # Validate max_matches if provided
    if max_matches is not None:
        max_matches = check_int_gt0(max_matches, "max_matches")
    # Validate max_dist if provided
    if max_dist is not None:
        if not isinstance(max_dist, (int, np.integer)):
            raise ValueError(f"max_dist must be int or None; got {type(max_dist)}")
        max_dist = int(max_dist)
        if max_dist < 0:
            raise ValueError(f"max_dist must be >= 0; got {max_dist}")

    # Compute Hamming distance matrix (Na, Nb)
    D = hamming_distance_matrix(descA, descB)

    # Read sizes
    Na, Nb = D.shape

    # Early exit
    if Na == 0 or Nb == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # For each A, find best B (smallest Hamming distance)
    ib_best = np.argmin(D, axis=1)

    # Best distances for each A
    d_best = D[np.arange(Na), ib_best].astype(np.float64, copy=False)

    # Apply distance threshold if requested
    if max_dist is None:
        keep = np.ones((Na,), dtype=bool)
    else:
        keep = (d_best <= float(max_dist))

    # Candidate matches
    ia = np.nonzero(keep)[0].astype(np.int64, copy=False)
    ib = ib_best[keep].astype(np.int64, copy=False)
    d1 = d_best[keep].astype(np.float64, copy=False)

    # If mutual requested, enforce B also picks A
    if mutual and ia.size > 0:
        ia_best_for_b = np.argmin(D, axis=0)
        ok = (ia == ia_best_for_b[ib])
        ia = ia[ok]
        ib = ib[ok]
        d1 = d1[ok]

    # If nothing remains
    if d1.size == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Convert to score where higher is better (negative distance)
    score = (-d1).astype(np.float64, copy=False)

    # Sort by score descending (smallest distance first)
    order = np.argsort(score)[::-1]
    ia = ia[order]
    ib = ib[order]
    score = score[order]

    # Truncate if requested
    if max_matches is not None:
        ia = ia[:max_matches]
        ib = ib[:max_matches]
        score = score[:max_matches]

    return MatchResult(ia=ia, ib=ib, score=score)


# Match packed binary descriptors using a Lowe-style ratio test (ORB-ish)
def match_descriptors_hamming_ratio(
    descA: np.ndarray,
    descB: np.ndarray,
    *,
    ratio: float = 0.8,
    mutual: bool = True,
    max_matches: int | None = None,
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
    D = hamming_distance_matrix(descA, descB)

    # Read sizes
    Na, Nb = D.shape

    # Early exit
    if Na == 0 or Nb == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # If only one candidate per row exists, ratio test is not defined
    if Nb < 2:
        # Best match per A
        j1 = np.argmin(D, axis=1)
        # Best distances
        d1 = D[np.arange(Na), j1].astype(np.float64, copy=False)
        # Keep all
        ia = np.arange(Na, dtype=np.int64)
        ib = j1.astype(np.int64, copy=False)
        # Score = negative distance (higher is better)
        score = (-d1).astype(np.float64, copy=False)

    else:
        # Get indices of the two nearest neighbours in B for each A
        idx2 = np.argpartition(D, kth=1, axis=1)[:, :2]

        # Gather candidate distances
        d2cand = D[np.arange(Na)[:, None], idx2].astype(np.float64, copy=False)

        # Sort the two candidates so col 0 is nearest
        order2 = np.argsort(d2cand, axis=1)
        idx_sorted = idx2[np.arange(Na)[:, None], order2]
        d_sorted = d2cand[np.arange(Na)[:, None], order2]

        # Nearest neighbour index and distance
        j1 = idx_sorted[:, 0]
        d1 = d_sorted[:, 0]

        # Second nearest distance
        d2 = d_sorted[:, 1]

        # Stabilise d2 (avoid divide-by-zero)
        d2 = np.maximum(d2, 1e-12)

        # Apply ratio test
        keep = (d1 / d2) < float(ratio)

        # Keep candidates
        ia = np.nonzero(keep)[0].astype(np.int64, copy=False)
        ib = j1[keep].astype(np.int64, copy=False)

        # Score = negative distance (higher is better)
        score = (-d1[keep]).astype(np.float64, copy=False)

    # Enforce mutual best if requested
    if mutual and score.size > 0:
        # Best A for each B is argmin over A
        ia_best_for_b = np.argmin(D, axis=0)
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

    # Sort by score descending (smallest distance first)
    order = np.argsort(score)[::-1]
    ia = ia[order]
    ib = ib[order]
    score = score[order]

    # Truncate if requested
    if max_matches is not None:
        ia = ia[:max_matches]
        ib = ib[:max_matches]
        score = score[:max_matches]

    return MatchResult(ia=ia, ib=ib, score=score)
