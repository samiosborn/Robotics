# scripts/diag_strict_on_loose_calibration.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg

from datasets.loader import load_sequence
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames


# Known refresh labels from single-frame counterfactual experiments
_REFRESH_LABELS: dict[str, dict[int, str]] = {
    "eth3d": {
        16: "load_bearing_refresh_bad_pose",
        17: "load_bearing_good_refresh",
        18: "mostly_neutral_refresh",
    },
    "kitti": {
        14: "load_bearing_good_refresh",
        16: "refresh_blocked_guard",
        17: "load_bearing_good_refresh",
        18: "refresh_blocked_guard",
        20: "refresh_blocked_guard",
    },
}

# Known pose quality labels
_POSE_LABELS: dict[str, dict[int, str]] = {
    "eth3d": {
        12: "bad_canonical_pose",
        16: "bad_canonical_pose_main_outlier",
    },
    "kitti": {},
}


def _replay_sequence(
    profile_path: Path,
    dataset_key: str,
    num_track: int,
) -> list[dict[str, Any]]:
    cfg, K = _load_runtime_cfg(profile_path)
    fkw = _frontend_kwargs_from_cfg(cfg)
    pnp_cfg = fkw["pnp_frontend_kwargs"]

    dataset_cfg = cfg["dataset"]
    run_cfg = cfg.get("run", {})
    dataset_name = str(dataset_cfg["name"])
    dataset_root = (ROOT / dataset_cfg["root"]).resolve()
    seq_name = str(dataset_cfg["seq"])

    seq = load_sequence(
        dataset_name,
        dataset_root,
        seq_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    run_bootstrap_cfg = run_cfg.get("bootstrap", {})
    i0 = int(run_bootstrap_cfg.get("i0", 0))
    i1 = int(run_bootstrap_cfg.get("i1", 1))

    im0, _, _ = seq.get(i0)
    im1, _, _ = seq.get(i1)
    boot = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=fkw["feature_cfg"],
        F_cfg=fkw["F_cfg"],
        H_cfg=fkw["H_cfg"],
        bootstrap_cfg=fkw["bootstrap_cfg"],
    )
    if not bool(boot.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed on {dataset_key}")

    seed = boot["seed"]
    events: list[dict[str, Any]] = []

    for step in range(num_track):
        frame_index = i1 + 1 + step
        if frame_index >= len(seq):
            break

        cur_im, _, _ = seq.get(frame_index)
        out = process_frame_against_seed(
            K, seed, cur_im,
            feature_cfg=fkw["feature_cfg"],
            F_cfg=fkw["F_cfg"],
            current_kf=frame_index,
            **pnp_cfg,
        )

        pipeline_ok = bool(out.get("ok", False))
        frame_stats = out.get("stats", {}) or {}
        pose_out = out.get("pose_out") or {}
        pose_stats = pose_out.get("stats", {}) or {}

        rescue_attempted = bool(pose_stats.get("pnp_support_rescue_attempted", False))
        rescue_succeeded = bool(pose_stats.get("pnp_support_rescue_succeeded", False))
        loose_fallback = bool(pose_stats.get("pnp_support_rescue_loose_localisation_fallback_succeeded", False))

        if rescue_succeeded:
            loose_inliers = int(pose_stats.get("pnp_support_rescue_loose_inliers", 0))
            subset_count = int(pose_stats.get("pnp_support_rescue_subset_count", 0))
            subset_strict = int(pose_stats.get("pnp_support_rescue_subset_strict_inliers", 0))
            loose_threshold = float(pose_stats.get("pnp_support_rescue_loose_threshold_px", 0.0))
            n_inliers = int(pose_stats.get("n_pnp_inliers", 0))
            cells = int(pose_stats.get("pnp_inlier_occupied_cells", 0))
            mcf = pose_stats.get("pnp_inlier_max_cell_fraction", None)

            refresh_triggered = bool(frame_stats.get("guarded_support_refresh_triggered", False))

            # strict-on-loose denominator: subset_count (re-evaluated loose inlier count)
            denom = int(subset_count) if int(subset_count) > 0 else int(loose_inliers)
            sol_count = int(subset_strict)
            sol_fraction = float(sol_count) / float(denom) if denom > 0 else None

            event: dict[str, Any] = {
                "dataset": dataset_key,
                "frame": frame_index,
                "pipeline_ok": pipeline_ok,
                "loose_threshold_px": loose_threshold,
                "loose_inliers": loose_inliers,
                "subset_count": subset_count,
                "strict_on_loose_count": sol_count,
                "strict_on_loose_fraction": sol_fraction,
                "is_loose_fallback": loose_fallback,
                "n_final_inliers": n_inliers,
                "cells": cells,
                "max_cell_fraction": mcf,
                "refresh_triggered": refresh_triggered,
                "refresh_label": _REFRESH_LABELS.get(dataset_key, {}).get(frame_index, "unclear"),
                "pose_label": _POSE_LABELS.get(dataset_key, {}).get(frame_index, "unknown"),
            }
            events.append(event)

        seed = out["seed"]

    return events


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "   —   "
    return f"{float(v):.{digits}f}"


def _print_table(events: list[dict[str, Any]]) -> None:
    header = (
        f"{'dataset':<8} {'frame':>5} {'thr':>5} "
        f"{'loose':>6} {'strict':>7} {'sol_n':>6} {'sol_frac':>9} "
        f"{'fallbk':>7} {'cells':>5} {'mcf':>6} {'refresh':>8} "
        f"{'refresh_label':<35} {'pose_label'}"
    )
    print(header)
    print("-" * len(header))
    for e in events:
        sol_n = int(e["strict_on_loose_count"])
        denom = int(e["subset_count"]) if int(e["subset_count"]) > 0 else int(e["loose_inliers"])
        sol_str = f"{sol_n:>3}/{denom:<3}"
        sol_frac = _fmt(e["strict_on_loose_fraction"])
        fallbk = "YES" if bool(e["is_loose_fallback"]) else "no"
        refresh = "YES" if bool(e["refresh_triggered"]) else "no"
        mcf = _fmt(e["max_cell_fraction"])
        print(
            f"{e['dataset']:<8} {e['frame']:>5} {e['loose_threshold_px']:>5.0f} "
            f"{e['loose_inliers']:>6} {e['n_final_inliers']:>7} {sol_str:>6} {sol_frac:>9} "
            f"{fallbk:>7} {e['cells']:>5} {mcf:>6} {refresh:>8} "
            f"{e['refresh_label']:<35} {e['pose_label']}"
        )


def _classify_signal(events: list[dict[str, Any]]) -> str:
    fallback_events = [e for e in events if bool(e["is_loose_fallback"])]
    non_fallback_events = [e for e in events if not bool(e["is_loose_fallback"])]

    n_fallback = len(fallback_events)
    n_non_fallback = len(non_fallback_events)

    fallback_bad = [e for e in fallback_events if "bad" in str(e["pose_label"]) or "bad" in str(e["refresh_label"])]
    non_fallback_bad = [e for e in non_fallback_events if "bad" in str(e["pose_label"]) or "bad" in str(e["refresh_label"])]

    print(f"\nSummary:")
    print(f"  total accepted rescues: {len(events)}")
    print(f"  loose fallback (strict-on-loose = 0): {n_fallback}")
    print(f"  strict rescue (strict-on-loose > 0): {n_non_fallback}")
    print(f"  fallback events with bad label: {len(fallback_bad)}")
    print(f"  non-fallback events with bad label: {len(non_fallback_bad)}")

    # Non-zero strict-on-loose values for strict rescues
    sol_fractions = [e["strict_on_loose_fraction"] for e in non_fallback_events if e["strict_on_loose_fraction"] is not None]
    if sol_fractions:
        print(f"  non-fallback strict-on-loose range: {min(sol_fractions):.3f} to {max(sol_fractions):.3f}")
        print(f"  non-fallback strict-on-loose median: {float(np.median(sol_fractions)):.3f}")

    if n_fallback == 0:
        return "inconclusive (no fallback events observed)"

    if n_fallback > 0 and len(fallback_bad) == n_fallback and len(non_fallback_bad) == 0:
        return "strict-on-loose zero is a strong pathological signal"

    if n_fallback > 0 and len(fallback_bad) == n_fallback and len(non_fallback_bad) > 0:
        return "strict-on-loose is a useful graded signal"

    if n_fallback > 0 and len(fallback_bad) < n_fallback:
        return "inconclusive (fallback events include non-bad cases)"

    return "inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth3d_profile", type=str,
                        default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--kitti_profile", type=str,
                        default=str(ROOT / "configs" / "profiles" / "kitti_odometry_00.yaml"))
    parser.add_argument("--eth3d_num_track", type=int, default=22)
    parser.add_argument("--kitti_num_track", type=int, default=20)
    args = parser.parse_args()

    print("=== Strict-on-loose calibration across ETH3D and KITTI ===\n")

    print(f"Running ETH3D cables_2_mono ({args.eth3d_num_track} frames)...")
    eth3d_events = _replay_sequence(
        Path(args.eth3d_profile).expanduser().resolve(),
        "eth3d",
        args.eth3d_num_track,
    )
    print(f"  accepted rescues: {len(eth3d_events)}")

    print(f"\nRunning KITTI sequence 00 ({args.kitti_num_track} frames)...")
    kitti_events = _replay_sequence(
        Path(args.kitti_profile).expanduser().resolve(),
        "kitti",
        args.kitti_num_track,
    )
    print(f"  accepted rescues: {len(kitti_events)}")

    all_events = eth3d_events + kitti_events

    print("\n=== Accepted loose rescue events ===\n")
    _print_table(all_events)

    print("\n=== ETH3D-only events ===\n")
    _print_table(eth3d_events)

    print("\n=== KITTI-only events ===\n")
    _print_table(kitti_events)

    classification = _classify_signal(all_events)

    print(f"\n=== Classification ===")
    print(f"  {classification}")


if __name__ == "__main__":
    main()
