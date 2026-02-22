# src/features/multiscale.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PIL import Image

from features.checks import check_finite_scalar
from features.checks import check_int_ge0
from features.checks import check_int_gt0
from features.checks import check_int_odd_ge1
from features.checks import check_keypoints_xy

from features.gradients import img_to_grey
from features.harris import harris_keypoints
from features.patches import extract_patches

from features.matching import MatchResult
from features.matching import hamming_distance_matrix

from features.descriptors import brief_make_pairs

from features.orientation import keypoint_orientations


# Bundle returned by multi-scale BRIEF extraction
@dataclass(frozen=True)
class MultiScaleBriefResult:
    # Keypoints in ORIGINAL image coordinates (x,y,score), shape (N,3)
    kps: np.ndarray
    # Pyramid level index per keypoint, shape (N,)
    level: np.ndarray
    # Scale factor of that level relative to original, shape (N,)
    scale: np.ndarray
    # Packed BRIEF descriptors, uint8, shape (N, nbytes)
    desc: np.ndarray
    # Optional orientations (radians), shape (N,) or None
    ori: np.ndarray | None


# Build a simple PIL image pyramid (level 0 = original)
def build_pil_pyramid(
    img: Image.Image,
    *,
    num_levels: int = 4,
    scale_factor: float = 0.75,
    min_size: int = 32,
) -> tuple[list[Image.Image], np.ndarray]:

    # --- Checks ---
    # Validate levels
    num_levels = check_int_gt0(num_levels, name="num_levels")
    # Validate scale factor
    scale_factor = check_finite_scalar(scale_factor, name="scale_factor")
    if scale_factor <= 0.0 or scale_factor >= 1.0:
        raise ValueError(f"scale_factor must be in (0,1); got {scale_factor}")
    # Validate min size
    min_size = check_int_gt0(min_size, name="min_size")

    # Ensure we have a PIL image
    if not isinstance(img, Image.Image):
        raise ValueError("img must be a PIL.Image")

    # Start pyramid with original
    pyr = [img]
    scales = [1.0]

    # Current image
    cur = img

    # Build downsampled levels
    for lvl in range(1, int(num_levels)):

        # Read current size
        W, H = cur.size

        # Compute next size
        Wn = int(round(float(W) * float(scale_factor)))
        Hn = int(round(float(H) * float(scale_factor)))

        # Stop if too small
        if Wn < int(min_size) or Hn < int(min_size):
            break

        # Downsample (bilinear is fine for now)
        nxt = cur.resize((Wn, Hn), resample=Image.BILINEAR)

        # Append
        pyr.append(nxt)
        scales.append(scales[-1] * float(scale_factor))

        # Step
        cur = nxt

    # Return pyramid and scales as array
    return pyr, np.asarray(scales, dtype=np.float64)


# Bilinear sampling on a patch tensor at floating coords (rs, cs), both shape (N,B)
def _bilinear_sample_patches(patches: np.ndarray, rs: np.ndarray, cs: np.ndarray) -> np.ndarray:

    # Read shapes
    N, P, _ = patches.shape

    # Convert coords to float arrays
    rs = np.asarray(rs, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)

    # Clip coords to valid bilinear range
    rmax = float(P - 1) - 1e-6
    cmax = float(P - 1) - 1e-6
    rs = np.clip(rs, 0.0, rmax)
    cs = np.clip(cs, 0.0, cmax)

    # Integer floors
    r0 = np.floor(rs).astype(np.int64)
    c0 = np.floor(cs).astype(np.int64)

    # Neighbour coords
    r1 = np.minimum(r0 + 1, P - 1)
    c1 = np.minimum(c0 + 1, P - 1)

    # Fractional parts
    dr = rs - r0
    dc = cs - c0

    # Build batch indices
    ii = np.arange(N, dtype=np.int64)[:, None]

    # Sample 4 corners
    v00 = patches[ii, r0, c0]
    v01 = patches[ii, r0, c1]
    v10 = patches[ii, r1, c0]
    v11 = patches[ii, r1, c1]

    # Bilinear interpolation
    v0 = v00 * (1.0 - dc) + v01 * dc
    v1 = v10 * (1.0 - dc) + v11 * dc
    v = v0 * (1.0 - dr) + v1 * dr

    return v


# Compute packed BRIEF descriptors from patches (no orientation)
def brief_from_patches_basic(
    patches: np.ndarray,
    pairs: np.ndarray,
    *,
    packbits: bool = True,
    bitorder: str = "little",
) -> np.ndarray:

    # Ensure arrays
    patches = np.asarray(patches)
    pairs = np.asarray(pairs)

    # Early exit
    if patches.shape[0] == 0:
        # Compute bytes even for empty
        n_bits = int(pairs.shape[0])
        n_bytes = int((n_bits + 7) // 8)
        return np.zeros((0, n_bytes), dtype=np.uint8)

    # Expect (N,P,P)
    if patches.ndim != 3:
        raise ValueError(f"patches must have shape (N,P,P); got {patches.shape}")

    # Expect (B,4) with rows/cols
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError(f"pairs must have shape (B,4); got {pairs.shape}")

    # Split coords as (r1,c1,r2,c2)
    r1 = pairs[:, 0].astype(np.int64, copy=False)
    c1 = pairs[:, 1].astype(np.int64, copy=False)
    r2 = pairs[:, 2].astype(np.int64, copy=False)
    c2 = pairs[:, 3].astype(np.int64, copy=False)

    # Gather intensities -> shape (N,B)
    a = patches[:, r1, c1]
    b = patches[:, r2, c2]

    # Bit is 1 if a < b
    bits = (a < b)

    # Return raw bits or packed bytes
    if not bool(packbits):
        return bits.astype(np.uint8)

    # Pack bits along last axis
    return np.packbits(bits, axis=1, bitorder=str(bitorder)).astype(np.uint8, copy=False)


# Compute packed BRIEF descriptors from patches using per-keypoint orientation (rotated sampling)
def brief_from_patches_oriented(
    patches: np.ndarray,
    pairs: np.ndarray,
    angles: np.ndarray,
    *,
    packbits: bool = True,
    bitorder: str = "little",
) -> np.ndarray:

    # Ensure arrays
    patches = np.asarray(patches)
    pairs = np.asarray(pairs)
    angles = np.asarray(angles, dtype=np.float64)

    # Early exit
    if patches.shape[0] == 0:
        n_bits = int(pairs.shape[0])
        n_bytes = int((n_bits + 7) // 8)
        return np.zeros((0, n_bytes), dtype=np.uint8)

    # Expect (N,P,P)
    if patches.ndim != 3:
        raise ValueError(f"patches must have shape (N,P,P); got {patches.shape}")

    # Expect (B,4)
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError(f"pairs must have shape (B,4); got {pairs.shape}")

    # Validate angles shape
    if angles.ndim != 1 or angles.shape[0] != patches.shape[0]:
        raise ValueError(f"angles must have shape (N,); got {angles.shape}")

    # Read shape
    N, P, _ = patches.shape

    # Patch centre in (row,col)
    cy = 0.5 * float(P - 1)
    cx = 0.5 * float(P - 1)

    # Split coords as (r1,c1,r2,c2)
    r1 = pairs[:, 0].astype(np.float64, copy=False)
    c1 = pairs[:, 1].astype(np.float64, copy=False)
    r2 = pairs[:, 2].astype(np.float64, copy=False)
    c2 = pairs[:, 3].astype(np.float64, copy=False)

    # Convert to offsets around centre in (x,y) where x=col, y=row
    x1 = c1 - cx
    y1 = r1 - cy
    x2 = c2 - cx
    y2 = r2 - cy

    # Precompute cos/sin per keypoint
    ca = np.cos(angles)[:, None]
    sa = np.sin(angles)[:, None]

    # Rotate offsets: [x';y'] = [c -s; s c] [x;y]
    x1p = ca * x1[None, :] - sa * y1[None, :]
    y1p = sa * x1[None, :] + ca * y1[None, :]

    x2p = ca * x2[None, :] - sa * y2[None, :]
    y2p = sa * x2[None, :] + ca * y2[None, :]

    # Convert back to patch coords (row,col)
    r1p = cy + y1p
    c1p = cx + x1p
    r2p = cy + y2p
    c2p = cx + x2p

    # Bilinear sample intensities at rotated coords
    a = _bilinear_sample_patches(patches, r1p, c1p)
    b = _bilinear_sample_patches(patches, r2p, c2p)

    # Bit is 1 if a < b
    bits = (a < b)

    # Return packed bytes
    if not bool(packbits):
        return bits.astype(np.uint8)

    # Pack bits along last axis
    return np.packbits(bits, axis=1, bitorder=str(bitorder)).astype(np.uint8, copy=False)


# Build multi-scale BRIEF features from an image
def build_multiscale_brief(
    img,
    cfg: dict,
    *,
    patch_size: int = 31,
    brief_bits: int = 256,
    brief_seed: int = 0,
    brief_orient: bool = False,
    num_levels: int = 4,
    scale_factor: float = 0.75,
    min_size: int = 32,
    max_kps_per_level: int | None = None,
) -> MultiScaleBriefResult:

    # --- Checks ---
    # Validate patch size
    patch_size = check_int_odd_ge1(patch_size, name="patch_size")
    # Validate brief bits
    brief_bits = check_int_gt0(brief_bits, name="brief_bits")
    # Validate num levels
    num_levels = check_int_gt0(num_levels, name="num_levels")
    # Validate scale factor
    scale_factor = check_finite_scalar(scale_factor, name="scale_factor")
    if scale_factor <= 0.0 or scale_factor >= 1.0:
        raise ValueError(f"scale_factor must be in (0,1); got {scale_factor}")
    # Validate min size
    min_size = check_int_gt0(min_size, name="min_size")
    # Validate per-level cap if provided
    if max_kps_per_level is not None:
        max_kps_per_level = check_int_gt0(max_kps_per_level, name="max_kps_per_level")

    # Convert to PIL if needed
    if isinstance(img, Image.Image):
        img_pil = img
    else:
        img_pil = Image.fromarray(np.asarray(img))

    # Build pyramid and scale list
    pyr, scales = build_pil_pyramid(
        img_pil,
        num_levels=int(num_levels),
        scale_factor=float(scale_factor),
        min_size=int(min_size),
    )

    # Build BRIEF pairs for the chosen patch size
    pairs = brief_make_pairs(int(patch_size), n_bits=int(brief_bits), seed=int(brief_seed))
    pairs = np.asarray(pairs)

    # Unpack image preprocessing config (match gradients.py)
    img_cfg = cfg["image"]
    eps = float(img_cfg.get("eps", 1e-8))
    dtype_name = img_cfg.get("dtype", "float64")
    dtype = getattr(np, dtype_name) if isinstance(dtype_name, str) else dtype_name
    luminance_weights = img_cfg["luminance_weights"]
    assume_srgb = bool(img_cfg.get("assume_srgb", True))
    normalise_01 = bool(img_cfg.get("normalise_01", True))
    eotf_params = img_cfg.get("eotf", {})

    # Prepare output collectors
    all_kps = []
    all_lvl = []
    all_scale = []
    all_desc = []
    all_ori = []

    # Loop pyramid levels
    for lvl, (im_lvl_pil, s) in enumerate(zip(pyr, scales)):

        # Detect Harris keypoints on this level
        h = harris_keypoints(im_lvl_pil, cfg)

        # Read keypoints array
        kps = np.asarray(h.kps)

        # Optional cap per level (use top scores as returned)
        if (max_kps_per_level is not None) and (kps.shape[0] > int(max_kps_per_level)):
            kps = kps[: int(max_kps_per_level), :]

        # Convert this level to greyscale float array
        im_lvl = img_to_grey(
            im_lvl_pil,
            luminance_weights=luminance_weights,
            eotf_params=eotf_params,
            assume_srgb=assume_srgb,
            normalise_01=normalise_01,
            dtype=dtype,
            eps=eps,
        )

        # Extract patches around keypoints (BRIEF wants raw intensities; do NOT z-score)
        pr = extract_patches(
            im_lvl,
            kps,
            patch_size=int(patch_size),
            border_margin=0,
            normalise=False,
            dtype=dtype,
            eps=eps,
        )

        # Skip if none survived
        if pr.patches.shape[0] == 0:
            continue

        # Compute orientations if requested (on this level image)
        if bool(brief_orient):
            ang = keypoint_orientations(
                im_lvl,
                pr.kps,
                sigma_d=1.0,
                truncate=3.0,
                window_radius=max(4, int(patch_size // 6)),
                dtype=np.float64,
                eps=float(eps),
            )
        else:
            ang = None

        # Compute descriptors (packed uint8)
        if ang is None:
            desc = brief_from_patches_basic(pr.patches, pairs, packbits=True, bitorder="little")
        else:
            desc = brief_from_patches_oriented(pr.patches, pairs, ang, packbits=True, bitorder="little")

        # Convert keypoints to ORIGINAL image coordinates
        # Level coords * (1/scale) maps back to original coords
        invs = 1.0 / float(s)
        kps_orig = np.asarray(pr.kps, dtype=np.float64).copy()
        kps_orig[:, 0] *= invs
        kps_orig[:, 1] *= invs

        # Record
        all_kps.append(kps_orig[:, :3])
        all_lvl.append(np.full((kps_orig.shape[0],), int(lvl), dtype=np.int64))
        all_scale.append(np.full((kps_orig.shape[0],), float(s), dtype=np.float64))
        all_desc.append(desc)

        # Record angles (or None)
        if ang is None:
            all_ori.append(np.full((kps_orig.shape[0],), 0.0, dtype=np.float64))
        else:
            all_ori.append(np.asarray(ang, dtype=np.float64))

    # If nothing found, return empties
    if len(all_desc) == 0:
        n_bytes = int((int(brief_bits) + 7) // 8)
        return MultiScaleBriefResult(
            kps=np.zeros((0, 3), dtype=np.float64),
            level=np.zeros((0,), dtype=np.int64),
            scale=np.zeros((0,), dtype=np.float64),
            desc=np.zeros((0, n_bytes), dtype=np.uint8),
            ori=np.zeros((0,), dtype=np.float64) if bool(brief_orient) else None,
        )

    # Concatenate collectors
    kps_all = np.vstack(all_kps)
    lvl_all = np.concatenate(all_lvl)
    sca_all = np.concatenate(all_scale)
    desc_all = np.vstack(all_desc)
    ori_all = np.concatenate(all_ori)

    # Return bundle
    return MultiScaleBriefResult(
        kps=kps_all,
        level=lvl_all,
        scale=sca_all,
        desc=desc_all,
        ori=ori_all if bool(brief_orient) else None,
    )


# Match BRIEF descriptors with a simple pyramid level gate (scale consistency)
def match_brief_hamming_with_scale_gate(
    descA: np.ndarray,
    levelA: np.ndarray,
    descB: np.ndarray,
    levelB: np.ndarray,
    *,
    mode: str = "nn",
    max_dist: int = 85,
    ratio: float = 0.8,
    scale_gate_levels: int = 1,
    mutual: bool = True,
    max_matches: int | None = None,
) -> MatchResult:

    # --- Checks ---
    # Require arrays
    descA = np.asarray(descA)
    descB = np.asarray(descB)
    levelA = np.asarray(levelA)
    levelB = np.asarray(levelB)

    # Validate descriptor dtypes
    if descA.dtype != np.uint8 or descB.dtype != np.uint8:
        raise ValueError("descA/descB must be uint8 packed BRIEF descriptors")

    # Validate dims
    if descA.ndim != 2 or descB.ndim != 2:
        raise ValueError(f"descA/descB must be 2D; got {descA.shape} and {descB.shape}")

    # Validate byte length match
    if descA.shape[1] != descB.shape[1]:
        raise ValueError(f"descriptor byte dims must match; got {descA.shape[1]} and {descB.shape[1]}")

    # Validate level shapes
    if levelA.ndim != 1 or levelA.shape[0] != descA.shape[0]:
        raise ValueError(f"levelA must be shape (Na,); got {levelA.shape}")
    if levelB.ndim != 1 or levelB.shape[0] != descB.shape[0]:
        raise ValueError(f"levelB must be shape (Nb,); got {levelB.shape}")

    # Validate mode
    mode = str(mode).lower()
    if mode not in {"nn", "ratio"}:
        raise ValueError("mode must be 'nn' or 'ratio'")

    # Validate ratio
    ratio = check_finite_scalar(ratio, name="ratio")
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"ratio must be in (0,1); got {ratio}")

    # Validate max dist
    max_dist = int(max_dist)
    if max_dist < 0:
        raise ValueError(f"max_dist must be >= 0; got {max_dist}")

    # Validate scale gate
    scale_gate_levels = check_int_ge0(scale_gate_levels, name="scale_gate_levels")

    # Validate max matches if provided
    if max_matches is not None:
        max_matches = check_int_gt0(max_matches, name="max_matches")

    # Read sizes
    Na = int(descA.shape[0])
    Nb = int(descB.shape[0])

    # Early exit
    if Na == 0 or Nb == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Compute full Hamming distance matrix (simple + fine for demo sizes)
    D = hamming_distance_matrix(descA, descB).astype(np.float64, copy=False)

    # Build a level-consistency mask: allow |lvlA - lvlB| <= gate
    # This broadcasts to (Na, Nb)
    gate = float(scale_gate_levels)
    M = (np.abs(levelA[:, None].astype(np.float64) - levelB[None, :].astype(np.float64)) <= gate)

    # Disallow mismatched scales by setting distance to +inf
    Dg = np.where(M, D, np.inf)

    # Allocate best / second-best distances and indices
    best_j = np.full((Na,), -1, dtype=np.int64)
    best_d = np.full((Na,), np.inf, dtype=np.float64)
    second_d = np.full((Na,), np.inf, dtype=np.float64)

    # For each A, find best and (optionally) second best among allowed Bs
    if Nb == 1:
        # Only one candidate exists
        best_j[:] = 0
        best_d[:] = Dg[:, 0]
    else:
        # Use argpartition to get two smallest distances per row
        idx2 = np.argpartition(Dg, kth=1, axis=1)[:, :2]
        d2 = Dg[np.arange(Na)[:, None], idx2]
        ord2 = np.argsort(d2, axis=1)
        idx_sorted = idx2[np.arange(Na)[:, None], ord2]
        d_sorted = d2[np.arange(Na)[:, None], ord2]
        best_j = idx_sorted[:, 0].astype(np.int64, copy=False)
        best_d = d_sorted[:, 0].astype(np.float64, copy=False)
        second_d = d_sorted[:, 1].astype(np.float64, copy=False)

    # Build keep mask depending on mode
    if mode == "nn":
        keep = (best_d <= float(max_dist))
    else:
        # Ratio test: d1 / d2 < ratio (guard d2)
        denom = np.maximum(second_d, 1e-12)
        keep = (best_d / denom) < float(ratio)

    # Candidate matches
    ia = np.nonzero(keep)[0].astype(np.int64, copy=False)
    ib = best_j[keep].astype(np.int64, copy=False)
    d1 = best_d[keep].astype(np.float64, copy=False)

    # If nothing, return empty
    if ia.size == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Mutual cross-check if requested
    if bool(mutual):
        # Best A per B under the same gating
        # Compute argmin over A for each B
        best_i_for_b = np.argmin(Dg, axis=0).astype(np.int64, copy=False)
        ok = (ia == best_i_for_b[ib])
        ia = ia[ok]
        ib = ib[ok]
        d1 = d1[ok]

    # If nothing after mutual
    if ia.size == 0:
        return MatchResult(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Score = negative distance so "higher is better"
    score = (-d1).astype(np.float64, copy=False)

    # Sort by score descending (i.e., smallest distance first)
    order = np.argsort(score)[::-1]
    ia = ia[order]
    ib = ib[order]
    score = score[order]

    # Truncate if requested
    if max_matches is not None:
        ia = ia[: int(max_matches)]
        ib = ib[: int(max_matches)]
        score = score[: int(max_matches)]

    return MatchResult(ia=ia, ib=ib, score=score)
