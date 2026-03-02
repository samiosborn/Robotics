# scripts/demo_track_two_frames.py

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.load_config import load_config

from features.gradients import img_to_grey
from features.harris import harris_keypoints
from features.patches import extract_patches

from features.matching import match_patches_ncc
from features.matching import match_brief_hamming_with_scale_gate
from features.descriptors import brief_make_pairs
from features.descriptors import brief_from_patches
from features.descriptors import brief_orientations_from_patches
from features.orientation import keypoint_orientations
from features.debug import print_brief_bitorder_checks
from features.debug import print_brief_hamming_diagnostics
from features.debug import print_brief_ransac_survivorship
from features.debug import print_brief_runtime_params
from features.debug import print_brief_unique_stats
from features.viz import draw_matches
from features.viz import draw_projected_box

from geometry.checks import as_2xN_points
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
    parser.add_argument("--patch", type=int, default=21)

    # NCC threshold
    parser.add_argument("--min_score", type=float, default=0.7)

    # BRIEF parameters
    parser.add_argument("--brief_bits", type=int, default=256)
    parser.add_argument("--brief_seed", type=int, default=0)

    # BRIEF orientation compensation (rotate BRIEF sampling pairs per keypoint)
    parser.add_argument("--brief_orient", action="store_true")
    parser.add_argument("--brief_orient_method", type=str, default="ic", choices=["ic", "grad"])

    # BRIEF matching controls (NN + max_dist baseline)
    parser.add_argument("--brief_max_dist", type=int, default=72)

    # BRIEF ratio test option (not default; usually stricter)
    parser.add_argument("--brief_mode", type=str, default="nn", choices=["nn", "ratio"])
    parser.add_argument("--brief_ratio", type=float, default=0.8)

    # Mutual cross-check
    parser.add_argument("--brief_mutual", dest="brief_mutual", action="store_true")
    parser.add_argument("--no_brief_mutual", dest="brief_mutual", action="store_false")
    parser.set_defaults(brief_mutual=True)

    # Multiscale BRIEF via pyramid
    parser.add_argument("--brief_multiscale", dest="brief_multiscale", action="store_true")
    parser.add_argument("--no_brief_multiscale", dest="brief_multiscale", action="store_false")
    parser.set_defaults(brief_multiscale=True)
    parser.add_argument("--pyr_levels", type=int, default=4)
    parser.add_argument("--pyr_scale", type=float, default=1.25)
    parser.add_argument("--scale_gate", type=int, default=3)

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
    parser.add_argument("--max_draw_inliers", type=int, default=200)

    # BRIEF diagnostics
    parser.add_argument("--debug_brief", action="store_true")

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
    out_matches_inliers = out_dir / f"{str(args.name)}_inliers.png"
    out_scene = out_dir / f"{str(args.name)}_scene.png"
    out_scene_inliers = out_dir / f"{str(args.name)}_scene_inliers.png"

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

    # Keep BRIEF descriptor packing consistent between both images.
    brief_bitorder = "little"

    # Optional BRIEF parameter echo for reproducibility.
    if bool(args.debug_brief) and str(args.match).lower() == "brief":
        print_brief_runtime_params(
            brief_bits=int(args.brief_bits),
            patch=int(args.patch),
            brief_orient=bool(args.brief_orient),
            multiscale=bool(args.brief_multiscale),
            pyr_levels=int(args.pyr_levels),
            pyr_scale=float(args.pyr_scale),
            scale_gate=int(args.scale_gate),
            brief_mode=str(args.brief_mode).lower().strip(),
            max_dist=None if args.brief_max_dist is None else int(args.brief_max_dist),
            mutual=bool(args.brief_mutual),
            ransac=bool(args.ransac),
            ransac_trials=int(args.ransac_trials),
            ransac_thresh=float(args.ransac_thresh),
            ransac_seed=int(args.ransac_seed),
            ransac_topk=int(args.ransac_topk),
        )
        print_brief_bitorder_checks(bitorder_a=brief_bitorder, bitorder_b=brief_bitorder)

    # Create per-level PIL images
    pyr0_imgs = []
    pyr1_imgs = []
    pyr_scales = []
    pyr_map0 = []
    pyr_map1 = []

    # Level 0 is original
    pyr0_imgs.append(img0)
    pyr1_imgs.append(img1)
    pyr_scales.append(1.0)
    pyr_map0.append((1.0, 1.0))
    pyr_map1.append((1.0, 1.0))

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
        # Exact mapping factors (account for rounding in resized dimensions)
        pyr_map0.append((float(W0) / float(w0), float(H0) / float(h0)))
        pyr_map1.append((float(W1) / float(w1), float(H1) / float(h1)))

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

            # Compute orientations if requested, otherwise keep axis-aligned BRIEF
            if bool(args.brief_orient):
                ori_method = str(args.brief_orient_method).lower().strip()
                if ori_method == "ic":
                    a0 = brief_orientations_from_patches(pr0.patches, dtype=np.float64)
                    a1 = brief_orientations_from_patches(pr1.patches, dtype=np.float64)
                elif ori_method == "grad":
                    a0 = keypoint_orientations(
                        im0_lvl,
                        pr0.kps,
                        sigma_d=1.0,
                        truncate=3.0,
                        window_radius=max(4, int(args.patch) // 2),
                        dtype=np.float64,
                        eps=float(eps),
                    )
                    a1 = keypoint_orientations(
                        im1_lvl,
                        pr1.kps,
                        sigma_d=1.0,
                        truncate=3.0,
                        window_radius=max(4, int(args.patch) // 2),
                        dtype=np.float64,
                        eps=float(eps),
                    )
                else:
                    raise ValueError(f"Unknown brief_orient_method '{ori_method}'")

                # Rotate BRIEF sampling pairs analytically per keypoint (no patch warping)
                d0 = brief_from_patches(pr0.patches, pairs, angles=a0, packbits=True, bitorder=brief_bitorder)
                d1 = brief_from_patches(pr1.patches, pairs, angles=a1, packbits=True, bitorder=brief_bitorder)

                # Optional orientation debug summary per level
                if bool(args.debug_brief):
                    print(
                        f"brief_diag[level={lvl}]: orient={ori_method} "
                        f"a0_med={float(np.median(a0)):.3f} a1_med={float(np.median(a1)):.3f}"
                    )
            else:
                d0 = brief_from_patches(pr0.patches, pairs, packbits=True, bitorder=brief_bitorder)
                d1 = brief_from_patches(pr1.patches, pairs, packbits=True, bitorder=brief_bitorder)

            # Map level keypoints back to base-image coordinates
            sx0, sy0 = pyr_map0[lvl]
            sx1, sy1 = pyr_map1[lvl]
            kps0_base = np.asarray(pr0.kps, dtype=np.float64).copy()
            kps1_base = np.asarray(pr1.kps, dtype=np.float64).copy()
            kps0_base[:, 0] *= float(sx0)
            kps0_base[:, 1] *= float(sy0)
            kps1_base[:, 0] *= float(sx1)
            kps1_base[:, 1] *= float(sy1)

            # Optional coordinate-mapping debug summary per level
            if bool(args.debug_brief):
                approx = float(pyr_scales[lvl])
                print(
                    f"brief_diag[level={lvl}]: map0=({sx0:.4f},{sy0:.4f}) map1=({sx1:.4f},{sy1:.4f}) "
                    f"approx_scalar={approx:.4f}"
                )

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

        # Optional BRIEF diagnostics before matching
        if bool(args.debug_brief):
            print_brief_unique_stats(descA, name="image0")
            print_brief_unique_stats(descB, name="image1")
            print_brief_hamming_diagnostics(
                descA,
                lvlA,
                descB,
                lvlB,
                mode=str(args.brief_mode).lower().strip(),
                max_dist=None if args.brief_max_dist is None else int(args.brief_max_dist),
                ratio=float(args.brief_ratio),
                mutual=bool(args.brief_mutual),
                scale_gate=int(scale_gate),
                n_bits=int(args.brief_bits),
            )

        # Run robust scale-gated BRIEF matching
        matches = match_brief_hamming_with_scale_gate(
            descA,
            lvlA,
            descB,
            lvlB,
            mode=str(args.brief_mode).lower(),
            max_dist=None if args.brief_max_dist is None else int(args.brief_max_dist),
            ratio=float(args.brief_ratio),
            mutual=bool(args.brief_mutual),
            max_matches=None if args.max_matches is None else int(args.max_matches),
            scale_gate=int(scale_gate),
        )
        ia = matches.ia
        ib = matches.ib
        score = matches.score

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

    # Keep the final pre-RANSAC match list for drawing.
    ia_draw = np.asarray(ia, dtype=np.int64)
    ib_draw = np.asarray(ib, dtype=np.int64)
    ransac_mask = None

    # Best homography (if any)
    H_best = None

    # RANSAC homography
    if bool(args.ransac) and ia_draw.size > 0:

        # Gather matched points (N,2)
        pts0 = np.asarray(kpsA[ia_draw, :2], dtype=np.float64)
        pts1 = np.asarray(kpsB[ib_draw, :2], dtype=np.float64)

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
            if bool(args.debug_brief) and str(args.match).lower() == "brief":
                print_brief_ransac_survivorship(mask=None, total=int(ia_draw.size))
        else:
            # Count inliers
            inliers = int(mask.sum())
            total = int(mask.size)
            print(f"ransac: inliers {inliers}/{total} (thresh={float(args.ransac_thresh):.2f}px)")
            ransac_mask = np.asarray(mask, dtype=bool)

            if bool(args.debug_brief) and str(args.match).lower() == "brief":
                print_brief_ransac_survivorship(mask=ransac_mask, total=int(ia_draw.size))

            # Refit H on all inliers
            if inliers >= 4:
                try:
                    H = estimate_homography(x0[:, ransac_mask], x1[:, ransac_mask], normalise=True)
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
    elif bool(args.ransac) and bool(args.debug_brief) and str(args.match).lower() == "brief":
        print_brief_ransac_survivorship(mask=None, total=int(ia_draw.size))

    # --------------------
    # Visualisation output
    # --------------------
    # Legacy-friendly default: when RANSAC succeeds, main output shows inliers.
    main_draw_inliers = bool(args.ransac) and (ransac_mask is not None)
    draw_matches(
        img0,
        img1,
        kpsA,
        kpsB,
        ia_draw,
        ib_draw,
        out_matches,
        max_draw=int(args.max_draw),
        draw_topk=int(args.max_draw),
        draw_inliers_only=main_draw_inliers,
        inlier_mask=ransac_mask if main_draw_inliers else None,
    )
    print(f"wrote: {out_matches}")

    if bool(args.ransac):
        draw_matches(
            img0,
            img1,
            kpsA,
            kpsB,
            ia_draw,
            ib_draw,
            out_matches_inliers,
            max_draw=int(args.max_draw_inliers),
            draw_topk=int(args.max_draw_inliers),
            draw_inliers_only=True,
            inlier_mask=ransac_mask,
        )
        print(f"wrote: {out_matches_inliers}")

    # If we have a homography, project img0 corners into img1 and draw the quad.
    if H_best is not None:
        draw_projected_box(img0, img1, H_best, out_scene, width=3)
        print(f"wrote: {out_scene}")
        if bool(args.ransac):
            draw_projected_box(img0, img1, H_best, out_scene_inliers, width=3)
            print(f"wrote: {out_scene_inliers}")
