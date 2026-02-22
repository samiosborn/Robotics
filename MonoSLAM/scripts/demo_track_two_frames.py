# scripts/demo_track_two_frames.py

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.load_config import load_config

from features.gradients import img_to_grey
from features.harris import harris_keypoints
from features.patches import extract_patches

from features.matching import match_patches_ncc
from features.matching import hamming_distance_matrix
from features.descriptors import brief_make_pairs
from features.descriptors import brief_from_patches

from geometry.checks import as_2xN_points
from geometry.homography import apply_homography
from geometry.homography import estimate_homography
from geometry.homography import estimate_homography_ransac


# Only run when executed as a script
if __name__ == "__main__":

    # Build argument parser
    parser = argparse.ArgumentParser()

    # Input images
    parser.add_argument("image0", type=str)
    parser.add_argument("image1", type=str)

    # Config path
    parser.add_argument("--cfg", type=str, default=str(ROOT / "src" / "config" / "features.yaml"))

    # Output directory and base name
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "out"))
    parser.add_argument("--name", type=str, default="matches")

    # Choose matching mode
    parser.add_argument("--match", type=str, default="ncc", choices=["ncc", "brief"])

    # Patch size (odd)
    parser.add_argument("--patch", type=int, default=11)

    # NCC threshold
    parser.add_argument("--min_score", type=float, default=0.7)

    # BRIEF parameters
    parser.add_argument("--brief_bits", type=int, default=256)
    parser.add_argument("--brief_seed", type=int, default=0)

    # BRIEF orientation (rotate patches into a canonical frame before BRIEF)
    parser.add_argument("--brief_orient", action="store_true")

    # BRIEF matching controls (NN + max_dist baseline)
    parser.add_argument("--brief_max_dist", type=int, default=85)

    # BRIEF ratio test option (not default; usually stricter)
    parser.add_argument("--brief_mode", type=str, default="nn", choices=["nn", "ratio"])
    parser.add_argument("--brief_ratio", type=float, default=0.8)

    # Mutual cross-check
    parser.add_argument("--brief_mutual", action="store_true")

    # Multiscale BRIEF via pyramid
    parser.add_argument("--brief_multiscale", action="store_true")
    parser.add_argument("--pyr_levels", type=int, default=4)
    parser.add_argument("--pyr_scale", type=float, default=1.25)
    parser.add_argument("--scale_gate", type=int, default=1)

    # Max matches to keep (pre-RANSAC)
    parser.add_argument("--max_matches", type=int, default=300)

    # RANSAC options
    parser.add_argument("--ransac", action="store_true")
    parser.add_argument("--ransac_trials", type=int, default=8000)
    parser.add_argument("--ransac_thresh", type=float, default=6.0)
    parser.add_argument("--ransac_seed", type=int, default=0)
    parser.add_argument("--ransac_topk", type=int, default=0)

    # Drawing options
    parser.add_argument("--max_draw", type=int, default=200)

    # Parse args
    args = parser.parse_args()

    # Resolve inputs
    p0 = Path(args.image0).expanduser().resolve()
    p1 = Path(args.image1).expanduser().resolve()
    pcfg = Path(args.cfg).expanduser().resolve()

    # Fail fast if inputs don't exist
    if not p0.exists():
        raise FileNotFoundError(f"image0 not found: {p0}")
    if not p1.exists():
        raise FileNotFoundError(f"image1 not found: {p1}")
    if not pcfg.exists():
        raise FileNotFoundError(f"config not found: {pcfg}")

    # Build output paths
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_matches = out_dir / f"{str(args.name)}.png"
    out_scene = out_dir / f"{str(args.name)}_scene.png"

    # Load images (force RGB to avoid palette / weird modes)
    img0 = Image.open(str(p0)).convert("RGB")
    img1 = Image.open(str(p1)).convert("RGB")

    # Load config dict
    cfg = load_config(pcfg)

    # Unpack image preprocessing config (match gradients.py behaviour)
    img_cfg = cfg["image"]
    eps = float(img_cfg.get("eps", 1e-8))
    dtype_name = img_cfg.get("dtype", "float64")
    dtype = getattr(np, dtype_name) if isinstance(dtype_name, str) else dtype_name
    luminance_weights = img_cfg["luminance_weights"]
    assume_srgb = bool(img_cfg.get("assume_srgb", True))
    normalise_01 = bool(img_cfg.get("normalise_01", True))
    eotf_params = img_cfg.get("eotf", {})

    # Convenience: the base image sizes
    W0, H0 = img0.size
    W1, H1 = img1.size

    # -----------------------------
    # Build image pyramids (optional)
    # -----------------------------

    # Decide number of levels
    if bool(args.brief_multiscale) and str(args.match).lower() == "brief":
        n_levels = int(args.pyr_levels)
    else:
        n_levels = 1

    # Validate pyramid params
    if n_levels < 1:
        raise ValueError(f"pyr_levels must be >= 1; got {n_levels}")
    pyr_scale = float(args.pyr_scale)
    if pyr_scale <= 1.0:
        raise ValueError(f"pyr_scale must be > 1.0; got {pyr_scale}")
    scale_gate = int(args.scale_gate)
    if scale_gate < 0:
        raise ValueError(f"scale_gate must be >= 0; got {scale_gate}")

    # Create per-level PIL images
    pyr0_imgs = []
    pyr1_imgs = []
    pyr_scales = []

    # Level 0 is original
    pyr0_imgs.append(img0)
    pyr1_imgs.append(img1)
    pyr_scales.append(1.0)

    # Build downsampled levels
    for lvl in range(1, n_levels):
        s = pyr_scale ** lvl
        w0 = max(8, int(round(W0 / s)))
        h0 = max(8, int(round(H0 / s)))
        w1 = max(8, int(round(W1 / s)))
        h1 = max(8, int(round(H1 / s)))
        pyr0_imgs.append(img0.resize((w0, h0), resample=Image.BILINEAR))
        pyr1_imgs.append(img1.resize((w1, h1), resample=Image.BILINEAR))
        pyr_scales.append(s)

    # ---------------------------------
    # Detect + describe across the pyramid
    # ---------------------------------

    # Storage for concatenated keypoints/descriptors
    all_kps0 = []
    all_kps1 = []
    all_lvl0 = []
    all_lvl1 = []
    all_desc0 = []
    all_desc1 = []

    # Storage for NCC-only path (single level)
    ncc_pr0 = None
    ncc_pr1 = None

    # Pre-build BRIEF pairs for the chosen patch size
    if str(args.match).lower() == "brief":
        pairs = brief_make_pairs(int(args.patch), n_bits=int(args.brief_bits), seed=int(args.brief_seed))

    # Loop pyramid levels
    for lvl in range(n_levels):

        # Read this level images
        I0_lvl = pyr0_imgs[lvl]
        I1_lvl = pyr1_imgs[lvl]

        # Convert to greyscale float arrays
        im0_lvl = img_to_grey(
            I0_lvl,
            luminance_weights=luminance_weights,
            eotf_params=eotf_params,
            assume_srgb=assume_srgb,
            normalise_01=normalise_01,
            dtype=dtype,
            eps=eps,
        )
        im1_lvl = img_to_grey(
            I1_lvl,
            luminance_weights=luminance_weights,
            eotf_params=eotf_params,
            assume_srgb=assume_srgb,
            normalise_01=normalise_01,
            dtype=dtype,
            eps=eps,
        )

        # Detect Harris keypoints on this level
        h0 = harris_keypoints(I0_lvl, cfg)
        h1 = harris_keypoints(I1_lvl, cfg)

        # Extract keypoints arrays
        kps0 = h0.kps
        kps1 = h1.kps

        # NCC path: only do level 0 (keep behaviour stable)
        if str(args.match).lower() == "ncc":
            if lvl != 0:
                continue

            # Print counts
            print(f"keypoints0: {kps0.shape[0]}")
            print(f"keypoints1: {kps1.shape[0]}")

            # Extract normalised patches around keypoints (aligned keypoints returned)
            ncc_pr0 = extract_patches(
                im0_lvl,
                kps0,
                patch_size=int(args.patch),
                border_margin=0,
                normalise=True,
                dtype=dtype,
                eps=eps,
            )
            ncc_pr1 = extract_patches(
                im1_lvl,
                kps1,
                patch_size=int(args.patch),
                border_margin=0,
                normalise=True,
                dtype=dtype,
                eps=eps,
            )

            # Print patch counts
            print(f"patches0: {ncc_pr0.patches.shape[0]}")
            print(f"patches1: {ncc_pr1.patches.shape[0]}")
            break

        # BRIEF path: extract raw patches (better for centroid-orientation + BRIEF comparisons)
        if str(args.match).lower() == "brief":

            # Extract raw patches around keypoints (no normalisation)
            pr0 = extract_patches(
                im0_lvl,
                kps0,
                patch_size=int(args.patch),
                border_margin=0,
                normalise=False,
                dtype=dtype,
                eps=eps,
            )
            pr1 = extract_patches(
                im1_lvl,
                kps1,
                patch_size=int(args.patch),
                border_margin=0,
                normalise=False,
                dtype=dtype,
                eps=eps,
            )

            # If there are no patches at this level, skip
            if pr0.patches.shape[0] == 0 or pr1.patches.shape[0] == 0:
                continue

            # Compute orientations from intensity centroid (ORB-style) if requested
            if bool(args.brief_orient):

                # Read patch size
                P = int(pr0.patches.shape[1])

                # Build centred coordinate grids (dx, dy)
                c = (P - 1) * 0.5
                xs = np.arange(P, dtype=np.float64) - c
                ys = np.arange(P, dtype=np.float64) - c
                dx = xs[None, :].repeat(P, axis=0)
                dy = ys[:, None].repeat(P, axis=1)

                # Compute angles for image0 patches
                a0 = []
                for i in range(pr0.patches.shape[0]):
                    patch = np.asarray(pr0.patches[i], dtype=np.float64)
                    m10 = float(np.sum(dx * patch))
                    m01 = float(np.sum(dy * patch))
                    a0.append(np.arctan2(m01, m10))
                a0 = np.asarray(a0, dtype=np.float64)

                # Compute angles for image1 patches
                a1 = []
                for i in range(pr1.patches.shape[0]):
                    patch = np.asarray(pr1.patches[i], dtype=np.float64)
                    m10 = float(np.sum(dx * patch))
                    m01 = float(np.sum(dy * patch))
                    a1.append(np.arctan2(m01, m10))
                a1 = np.asarray(a1, dtype=np.float64)

                # Rotate patches into canonical orientation (rotate by -angle)
                rp0 = np.empty_like(pr0.patches, dtype=np.float64)
                rp1 = np.empty_like(pr1.patches, dtype=np.float64)

                # Rotate patches for image0
                for i in range(pr0.patches.shape[0]):
                    patch = np.asarray(pr0.patches[i], dtype=np.float64)
                    patch_u8 = np.clip(patch, 0.0, 1.0)
                    patch_u8 = (patch_u8 * 255.0).round().astype(np.uint8)
                    pilp = Image.fromarray(patch_u8, mode="L")
                    deg = float(-a0[i] * 180.0 / np.pi)
                    pilr = pilp.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=0)
                    arr = np.asarray(pilr, dtype=np.float64) / 255.0
                    rp0[i] = arr

                # Rotate patches for image1
                for i in range(pr1.patches.shape[0]):
                    patch = np.asarray(pr1.patches[i], dtype=np.float64)
                    patch_u8 = np.clip(patch, 0.0, 1.0)
                    patch_u8 = (patch_u8 * 255.0).round().astype(np.uint8)
                    pilp = Image.fromarray(patch_u8, mode="L")
                    deg = float(-a1[i] * 180.0 / np.pi)
                    pilr = pilp.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=0)
                    arr = np.asarray(pilr, dtype=np.float64) / 255.0
                    rp1[i] = arr

                # Use rotated patches for BRIEF
                use_patches0 = rp0
                use_patches1 = rp1

            else:
                # Use raw patches as-is
                use_patches0 = pr0.patches
                use_patches1 = pr1.patches

            # Compute packed BRIEF descriptors from patches
            d0 = brief_from_patches(use_patches0, pairs, packbits=True, bitorder="little")
            d1 = brief_from_patches(use_patches1, pairs, packbits=True, bitorder="little")

            # Map level keypoints back to base-image coordinates
            s = float(pyr_scales[lvl])
            kps0_base = np.asarray(pr0.kps, dtype=np.float64).copy()
            kps1_base = np.asarray(pr1.kps, dtype=np.float64).copy()
            kps0_base[:, 0] *= s
            kps0_base[:, 1] *= s
            kps1_base[:, 0] *= s
            kps1_base[:, 1] *= s

            # Append into global lists
            all_kps0.append(kps0_base)
            all_kps1.append(kps1_base)
            all_desc0.append(d0)
            all_desc1.append(d1)
            all_lvl0.append(np.full((d0.shape[0],), lvl, dtype=np.int32))
            all_lvl1.append(np.full((d1.shape[0],), lvl, dtype=np.int32))

    # --------------------
    # Matching and RANSAC
    # --------------------

    # Match results holder
    ia = None
    ib = None
    score = None

    # NCC path
    if str(args.match).lower() == "ncc":

        # Run NCC matching on normalised patches
        matches = match_patches_ncc(
            ncc_pr0.patches,
            ncc_pr1.patches,
            min_score=float(args.min_score),
            mutual=True,
            max_matches=int(args.max_matches),
            eps=eps,
            dtype=dtype,
            assume_normalised=True,
        )

        # Report
        print(f"matches: {matches.ia.shape[0]}")
        if matches.score.size > 0:
            print(f"top_score: {float(matches.score[0]):.3f}")
            print(f"mean_score: {float(matches.score.mean()):.3f}")

        # Use patch-aligned keypoints
        kpsA = ncc_pr0.kps
        kpsB = ncc_pr1.kps

        # Start with all matches
        ia = matches.ia.copy()
        ib = matches.ib.copy()
        score = matches.score.copy()

    # BRIEF path (multiscale capable)
    if str(args.match).lower() == "brief":

        # If nothing was collected, stop early
        if len(all_desc0) == 0 or len(all_desc1) == 0:
            print("matches: 0")
            canvas = Image.new("RGB", (img0.size[0] + img1.size[0], max(img0.size[1], img1.size[1])))
            canvas.paste(img0.convert("RGB"), (0, 0))
            canvas.paste(img1.convert("RGB"), (img0.size[0], 0))
            canvas.save(str(out_matches))
            print(f"wrote: {out_matches}")
            raise SystemExit(0)

        # Concatenate across levels
        descA = np.concatenate(all_desc0, axis=0)
        descB = np.concatenate(all_desc1, axis=0)
        kpsA = np.concatenate(all_kps0, axis=0)
        kpsB = np.concatenate(all_kps1, axis=0)
        lvlA = np.concatenate(all_lvl0, axis=0)
        lvlB = np.concatenate(all_lvl1, axis=0)

        # Print counts
        print(f"keypoints0: {int(np.sum([x.shape[0] for x in all_kps0]))}")
        print(f"keypoints1: {int(np.sum([x.shape[0] for x in all_kps1]))}")
        print(f"patches0: {descA.shape[0]}")
        print(f"patches1: {descB.shape[0]}")

        # Build per-level index lists for B (for scale gating)
        max_lvl = int(max(int(lvlA.max()), int(lvlB.max())))
        idxB_by_lvl = []
        for l in range(max_lvl + 1):
            idxB_by_lvl.append(np.nonzero(lvlB == l)[0].astype(np.int64, copy=False))

        # Prepare arrays for best match per A
        best_ib = np.full((descA.shape[0],), -1, dtype=np.int64)
        best_d = np.full((descA.shape[0],), np.inf, dtype=np.float64)
        best_lB = np.full((descA.shape[0],), -1, dtype=np.int32)

        # For each level in A, compare to gated levels in B
        for lA in range(max_lvl + 1):

            # Indices of A at this level
            idxA = np.nonzero(lvlA == lA)[0].astype(np.int64, copy=False)
            if idxA.size == 0:
                continue

            # Build allowed B levels
            lmin = max(0, lA - int(scale_gate))
            lmax = min(max_lvl, lA + int(scale_gate))

            # Loop allowed B levels
            for lB in range(lmin, lmax + 1):

                # Indices of B at this level
                idxB = idxB_by_lvl[lB]
                if idxB.size == 0:
                    continue

                # Compute Hamming distances for this (A_level, B_level) block
                D = hamming_distance_matrix(descA[idxA], descB[idxB]).astype(np.float64, copy=False)

                # Best B in this block for each A row
                j = np.argmin(D, axis=1)
                d = D[np.arange(D.shape[0]), j]

                # Update global best if improved
                better = d < best_d[idxA]
                if np.any(better):
                    a_up = idxA[better]
                    best_d[a_up] = d[better]
                    best_ib[a_up] = idxB[j[better]]
                    best_lB[a_up] = int(lB)

        # Apply NN distance gate
        if args.brief_max_dist is None:
            keep = best_ib >= 0
        else:
            keep = (best_ib >= 0) & (best_d <= float(int(args.brief_max_dist)))

        # Candidate matches
        ia = np.nonzero(keep)[0].astype(np.int64, copy=False)
        ib = best_ib[keep].astype(np.int64, copy=False)
        d1 = best_d[keep].astype(np.float64, copy=False)

        # Optional ratio test (compute 2nd best within the same gating)
        if str(args.brief_mode).lower() == "ratio":

            # If we can't define a ratio (not enough candidates), fall back to NN keep
            ratio = float(args.brief_ratio)
            if ratio <= 0.0 or ratio >= 1.0:
                raise ValueError(f"brief_ratio must be in (0,1); got {ratio}")

            # Build second-best distances
            second_d = np.full((descA.shape[0],), np.inf, dtype=np.float64)

            # For each level in A, compute second best among gated B by brute force on blocks
            for lA in range(max_lvl + 1):
                idxA = np.nonzero(lvlA == lA)[0].astype(np.int64, copy=False)
                if idxA.size == 0:
                    continue
                lmin = max(0, lA - int(scale_gate))
                lmax = min(max_lvl, lA + int(scale_gate))
                for lB in range(lmin, lmax + 1):
                    idxB = idxB_by_lvl[lB]
                    if idxB.size == 0:
                        continue
                    D = hamming_distance_matrix(descA[idxA], descB[idxB]).astype(np.float64, copy=False)
                    if D.shape[1] < 2:
                        continue
                    # Two smallest per row via partition
                    two = np.partition(D, kth=1, axis=1)[:, :2]
                    # Sort the two values to ensure [0] is best, [1] is second
                    two.sort(axis=1)
                    # Update second best (global) conservatively
                    # Note: this is approximate if best and second-best come from different blocks; good enough for demo
                    second_d[idxA] = np.minimum(second_d[idxA], two[:, 1])

            # Apply ratio test on current candidates
            d2 = second_d[ia]
            d2 = np.maximum(d2, 1.0)
            keep_ratio = (d1 / d2) < float(ratio)
            ia = ia[keep_ratio]
            ib = ib[keep_ratio]
            d1 = d1[keep_ratio]

        # Mutual cross-check (keep only best A for each B among kept pairs)
        if bool(args.brief_mutual) and ia.size > 0:

            # Sort by (ib asc, distance asc)
            order = np.lexsort((d1, ib))
            ia_s = ia[order]
            ib_s = ib[order]
            d_s = d1[order]

            # Keep first occurrence per ib (smallest distance)
            keep_m = np.ones((ib_s.size,), dtype=bool)
            keep_m[1:] = ib_s[1:] != ib_s[:-1]

            # Apply
            ia = ia_s[keep_m]
            ib = ib_s[keep_m]
            d1 = d_s[keep_m]

        # Convert to score where higher is better
        score = (-d1).astype(np.float64, copy=False)

        # Sort by score descending (i.e., smallest distance first)
        order = np.argsort(score)[::-1]
        ia = ia[order]
        ib = ib[order]
        score = score[order]

        # Truncate to max_matches
        if args.max_matches is not None:
            ia = ia[: int(args.max_matches)]
            ib = ib[: int(args.max_matches)]
            score = score[: int(args.max_matches)]

        # Report
        print(f"matches: {ia.shape[0]}")
        if score.size > 0:
            print(f"top_score: {float(score[0]):.3f}")
            print(f"mean_score: {float(score.mean()):.3f}")

    # Optionally take only top-K matches before RANSAC
    if bool(args.ransac) and int(args.ransac_topk) > 0 and ia.size > 0:
        k = int(args.ransac_topk)
        ia = ia[:k]
        ib = ib[:k]
        score = score[:k]

    # Default: keep all matches for drawing
    ia_keep = ia
    ib_keep = ib

    # Best homography (if any)
    H_best = None

    # RANSAC homography
    if bool(args.ransac) and ia_keep.size > 0:

        # Gather matched points (N,2)
        pts0 = np.asarray(kpsA[ia_keep, :2], dtype=np.float64)
        pts1 = np.asarray(kpsB[ib_keep, :2], dtype=np.float64)

        # Convert to geometry convention (2,N)
        x0 = as_2xN_points(pts0, name="x0", finite=True)
        x1 = as_2xN_points(pts1, name="x1", finite=True)

        # Run RANSAC
        H, mask, reason = estimate_homography_ransac(
            x0,
            x1,
            num_trials=int(args.ransac_trials),
            threshold=float(args.ransac_thresh),
            normalise=True,
            seed=int(args.ransac_seed),
        )

        # Handle failure
        if mask is None or reason is not None:
            print(f"ransac: failed ({reason})")
        else:
            # Count inliers
            inliers = int(mask.sum())
            total = int(mask.size)
            print(f"ransac: inliers {inliers}/{total} (thresh={float(args.ransac_thresh):.2f}px)")

            # Filter to inliers
            ia_keep = ia_keep[mask]
            ib_keep = ib_keep[mask]

            # Refit H on all inliers
            if inliers >= 4:
                try:
                    H = estimate_homography(x0[:, mask], x1[:, mask], normalise=True)
                except Exception:
                    pass

            # Store
            H_best = H

            # Print H normalised
            if H_best is not None:
                s = float(H_best[2, 2]) if abs(float(H_best[2, 2])) > 1e-12 else 1.0
                Hn = H_best / s
                print("ransac: H (normalised):")
                print(Hn)

    # --------------------
    # Visualisation output
    # --------------------

    # Prepare side-by-side canvas
    A = img0.convert("RGB")
    B = img1.convert("RGB")
    WA, HA = A.size
    WB, HB = B.size
    canvas = Image.new("RGB", (WA + WB, max(HA, HB)))
    canvas.paste(A, (0, 0))
    canvas.paste(B, (WA, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw at most max_draw matches
    M = int(min(int(ia_keep.size), int(args.max_draw)))
    for m in range(M):

        # Read indices
        i = int(ia_keep[m])
        j = int(ib_keep[m])

        # Read points
        x0p = float(kpsA[i, 0])
        y0p = float(kpsA[i, 1])
        x1p = float(kpsB[j, 0]) + float(WA)
        y1p = float(kpsB[j, 1])

        # Draw circles
        r = 3
        draw.ellipse((x0p - r, y0p - r, x0p + r, y0p + r))
        draw.ellipse((x1p - r, y1p - r, x1p + r, y1p + r))

        # Draw line
        draw.line((x0p, y0p, x1p, y1p), width=1)

    # Save match visual
    canvas.save(str(out_matches))
    print(f"wrote: {out_matches}")

    # If we have a homography, project img0 corners into img1 and draw the quad
    if H_best is not None:

        # Build source corners in image0 coords (2,4)
        x = np.array(
            [
                [0.0, float(W0 - 1), float(W0 - 1), 0.0],
                [0.0, 0.0, float(H0 - 1), float(H0 - 1)],
            ],
            dtype=float,
        )

        # Project corners into scene
        y = apply_homography(H_best, x)

        # Convert into PIL polygon list
        quad = [(float(y[0, i]), float(y[1, i])) for i in range(y.shape[1])]

        # Draw onto scene image
        scene = img1.convert("RGB")
        dscene = ImageDraw.Draw(scene)
        dscene.polygon(quad, outline=(255, 0, 0), width=3)

        # Save
        scene.save(str(out_scene))
        print(f"wrote: {out_scene}")
