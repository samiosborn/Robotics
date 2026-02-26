# src/features/debug.py

from __future__ import annotations

import numpy as np

from features.matching import hamming_distance_matrix


# Print the key BRIEF-related runtime parameters used by the demo.
def print_brief_runtime_params(
    *,
    brief_bits: int,
    patch: int,
    brief_orient: bool,
    multiscale: bool,
    pyr_levels: int,
    pyr_scale: float,
    scale_gate: int,
    brief_mode: str,
    max_dist: int | None,
    mutual: bool,
    ransac: bool,
    ransac_trials: int,
    ransac_thresh: float,
    ransac_seed: int,
    ransac_topk: int,
) -> None:
    print(
        "brief_debug[params]: "
        f"brief_bits={int(brief_bits)} "
        f"patch={int(patch)} "
        f"brief_orient={bool(brief_orient)} "
        f"multiscale={bool(multiscale)} "
        f"pyr_levels={int(pyr_levels)} "
        f"pyr_scale={float(pyr_scale):.6g} "
        f"scale_gate={int(scale_gate)} "
        f"brief_mode={str(brief_mode)} "
        f"max_dist={max_dist} "
        f"mutual={bool(mutual)} "
        f"ransac={bool(ransac)} "
        f"ransac_trials={int(ransac_trials)} "
        f"ransac_thresh={float(ransac_thresh):.6g} "
        f"ransac_seed={int(ransac_seed)} "
        f"ransac_topk={int(ransac_topk)}"
    )


# Print descriptor uniqueness diagnostics and warn on suspicious duplication.
def print_brief_unique_stats(desc: np.ndarray, *, name: str, warn_unique_ratio: float = 0.5) -> None:
    if not isinstance(desc, np.ndarray) or desc.ndim != 2:
        print(f"brief_debug[unique:{name}]: invalid descriptor array")
        return
    total = int(desc.shape[0])
    if total == 0:
        print(f"brief_debug[unique:{name}]: total=0 unique_count=0 unique_ratio=0.000")
        return
    unique_count = int(np.unique(desc, axis=0).shape[0])
    unique_ratio = float(unique_count / max(total, 1))
    print(
        f"brief_debug[unique:{name}]: total={total} "
        f"unique_count={unique_count} unique_ratio={unique_ratio:.3f}"
    )
    if unique_ratio < float(warn_unique_ratio):
        print(
            f"brief_debug[warn]: low unique_ratio for {name} "
            f"({unique_ratio:.3f} < {float(warn_unique_ratio):.3f})"
        )


# Warn if descriptor bit packing order could be inconsistent across images.
def print_brief_bitorder_checks(*, bitorder_a: str, bitorder_b: str) -> None:
    ba = str(bitorder_a).lower().strip()
    bb = str(bitorder_b).lower().strip()
    print(
        "brief_debug[bitorder]: "
        f"image0={ba} image1={bb} matcher=xor_popcount(bytes)"
    )
    if ba not in {"little", "big"} or bb not in {"little", "big"}:
        print("brief_debug[warn]: unexpected bitorder label; expected 'little' or 'big'")
    if ba != bb:
        print("brief_debug[warn]: BRIEF bitorder mismatch between images can corrupt Hamming matching")


def _format_percentiles(values: np.ndarray, percents: np.ndarray) -> str:
    parts = []
    for p, v in zip(percents.tolist(), values.tolist()):
        parts.append(f"p{int(p)}={float(v):.1f}")
    return " ".join(parts)


# Print BRIEF Hamming diagnostics using the same gating/mutual logic as matching.
def print_brief_hamming_diagnostics(
    descA: np.ndarray,
    levelsA: np.ndarray,
    descB: np.ndarray,
    levelsB: np.ndarray,
    *,
    n_bits: int,
    mode: str = "nn",
    max_dist: int | None = None,
    ratio: float = 0.8,
    mutual: bool = True,
    scale_gate: int = 1,
) -> None:
    # --- Checks ---
    if not isinstance(descA, np.ndarray) or not isinstance(descB, np.ndarray):
        print("brief_debug[hamming]: invalid descriptors")
        return
    if descA.ndim != 2 or descB.ndim != 2:
        print("brief_debug[hamming]: descriptors must be 2D")
        return
    if descA.shape[0] == 0 or descB.shape[0] == 0:
        print("brief_debug[hamming]: empty descriptors")
        return
    if descA.dtype != np.uint8 or descB.dtype != np.uint8:
        print("brief_debug[hamming]: descriptors must be packed uint8")
        return

    # Distance matrix computed once and reused for all statistics.
    D = hamming_distance_matrix(descA, descB).astype(np.float64, copy=False)
    levelsA = np.asarray(levelsA)
    levelsB = np.asarray(levelsB)
    if levelsA.ndim != 1 or levelsA.shape[0] != descA.shape[0]:
        print("brief_debug[hamming]: invalid levelsA shape")
        return
    if levelsB.ndim != 1 or levelsB.shape[0] != descB.shape[0]:
        print("brief_debug[hamming]: invalid levelsB shape")
        return

    # Level gating matches match_brief_hamming_with_scale_gate behavior.
    M = (
        np.abs(levelsA[:, None].astype(np.int64) - levelsB[None, :].astype(np.int64))
        <= int(scale_gate)
    )
    Dg = np.where(M, D, np.inf)
    Na, Nb = Dg.shape

    best_j = np.full((Na,), -1, dtype=np.int64)
    best_d = np.full((Na,), np.inf, dtype=np.float64)
    second_d = np.full((Na,), np.inf, dtype=np.float64)
    valid_row = np.isfinite(Dg).any(axis=1)

    if np.any(valid_row):
        Dv = Dg[valid_row]
        j = np.argmin(Dv, axis=1)
        d = Dv[np.arange(Dv.shape[0]), j]
        best_j[valid_row] = j.astype(np.int64, copy=False)
        best_d[valid_row] = d.astype(np.float64, copy=False)
        if Nb >= 2:
            two = np.partition(Dv, kth=1, axis=1)[:, :2]
            two.sort(axis=1)
            second_d[valid_row] = two[:, 1]

    d1 = best_d[valid_row]
    if d1.size == 0:
        print("brief_debug[hamming]: no valid candidates after scale gating")
        return

    # Required percentiles for best distance.
    percents = np.array([0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100], dtype=np.float64)
    pvals = np.percentile(d1, percents)
    print(
        "brief_debug[hamming]: best_d_percentiles "
        + _format_percentiles(pvals, percents)
        + f" (valid_rows={int(valid_row.sum())}/{int(Na)})"
    )

    p50 = float(np.percentile(d1, 50.0))
    n_bits_f = float(n_bits)
    if p50 > (0.45 * n_bits_f):
        print(
            f"brief_debug[warn]: p50(best_d) is high ({p50:.1f} > {0.45 * n_bits_f:.1f}); "
            "matches may be weak/random"
        )
    if p50 < (0.05 * n_bits_f):
        print(
            f"brief_debug[warn]: p50(best_d) is very low ({p50:.1f} < {0.05 * n_bits_f:.1f}); "
            "descriptors may be over-correlated/duplicated"
        )

    # Survivorship counts with the same ordering as matcher.
    after_max_dist = valid_row.copy()
    if max_dist is not None:
        after_max_dist = after_max_dist & (best_d <= float(max_dist))
    count_after_max_dist = int(after_max_dist.sum())
    print(f"brief_debug[survivorship]: after_max_dist={count_after_max_dist}")

    keep = after_max_dist.copy()
    mode = str(mode).lower().strip()
    if mode == "ratio":
        denom = np.maximum(second_d, 1e-12)
        ratio_ok = np.isfinite(second_d) & ((best_d / denom) < float(ratio))
        keep = keep & ratio_ok
        print(f"brief_debug[survivorship]: after_ratio={int(keep.sum())}")

    ia = np.nonzero(keep)[0].astype(np.int64, copy=False)
    ib = best_j[keep].astype(np.int64, copy=False)

    if bool(mutual) and ia.size > 0:
        valid_col = np.isfinite(Dg).any(axis=0)
        best_i_for_b = np.full((Nb,), -1, dtype=np.int64)
        if np.any(valid_col):
            Dc = Dg[:, valid_col]
            best_i_for_b[valid_col] = np.argmin(Dc, axis=0).astype(np.int64, copy=False)
        ok = ia == best_i_for_b[ib]
        ia = ia[ok]

    print(f"brief_debug[survivorship]: after_mutual={int(ia.size)}")


# Print final RANSAC inlier survivorship for BRIEF matches.
def print_brief_ransac_survivorship(*, mask: np.ndarray | None, total: int) -> None:
    total = int(total)
    if total <= 0:
        print("brief_debug[survivorship]: after_ransac=0/0")
        return
    if mask is None:
        print(f"brief_debug[survivorship]: after_ransac=0/{total} (ransac_failed)")
        return
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1 or m.shape[0] != total:
        print(f"brief_debug[survivorship]: after_ransac=?/{total} (mask_shape_mismatch)")
        return
    print(f"brief_debug[survivorship]: after_ransac={int(m.sum())}/{total}")
