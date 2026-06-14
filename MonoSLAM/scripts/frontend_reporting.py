from __future__ import annotations

STANDARD_FRAME_STAT_FIELDS = (
    "frame_index",
    "reference_keyframe_index",
    "pipeline_ok",
    "pipeline_reason",
    "n_track_inliers",
    "n_pnp_corr",
    "n_pnp_inliers",
    "n_new_added",
    "pipeline_keyframe_promoted",
    "seed_landmarks_before",
    "seed_landmarks_after",
    "rescue_attempted",
    "rescue_succeeded",
    "localisation_only_rescue_frame",
    "support_refresh_triggered",
    "local_ba_attempted",
    "local_ba_skipped",
    "local_ba_succeeded",
    "local_ba_reason",
    "local_ba_skip_reason",
    "local_ba_n_local_keyframes",
    "local_ba_n_local_landmarks",
    "local_ba_n_observations",
    "local_ba_initial_mean_reproj_error_px",
    "local_ba_initial_median_reproj_error_px",
    "local_ba_final_mean_reproj_error_px",
    "local_ba_final_median_reproj_error_px",
    "local_ba_iterations",
    "local_ba_accepted_iterations",
    "local_ba_initial_damping",
    "local_ba_final_damping",
)


# Count landmarks in a seed dictionary
def seed_landmark_count(seed: dict | None) -> int:
    if not isinstance(seed, dict):
        return 0

    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        return 0

    return int(len(landmarks))


# Build the standard per-frame diagnostic fields
def standard_frame_stats(
    *,
    frame_index: int,
    reference_keyframe_index: int | None,
    frontend_out: dict | None = None,
    stats: dict | None = None,
    seed_before: dict | None = None,
    seed_after: dict | None = None,
    seed_landmarks_before: int | None = None,
    seed_landmarks_after: int | None = None,
) -> dict:
    frontend_out = frontend_out if isinstance(frontend_out, dict) else {}
    stats = stats if isinstance(stats, dict) else frontend_out.get("stats", {})
    stats = stats if isinstance(stats, dict) else {}

    if seed_landmarks_before is None:
        seed_landmarks_before = seed_landmark_count(seed_before)

    if seed_landmarks_after is None:
        if "seed_landmarks_after" in stats:
            seed_landmarks_after = int(stats.get("seed_landmarks_after", 0))
        else:
            seed_landmarks_after = seed_landmark_count(
                seed_after if seed_after is not None else frontend_out.get("seed", {})
            )

    return {
        "frame_index": int(frame_index),
        "reference_keyframe_index": None if reference_keyframe_index is None else int(reference_keyframe_index),
        "pipeline_ok": bool(frontend_out.get("ok", stats.get("ok", False))),
        "pipeline_reason": stats.get("reason", None),
        "n_track_inliers": int(stats.get("n_track_inliers", 0)),
        "n_pnp_corr": int(stats.get("n_pnp_corr", 0)),
        "n_pnp_inliers": int(stats.get("n_pnp_inliers", 0)),
        "n_new_added": int(stats.get("n_new_added", 0)),
        "pipeline_keyframe_promoted": bool(stats.get("keyframe_promoted", stats.get("pipeline_keyframe_promoted", False))),
        "seed_landmarks_before": int(seed_landmarks_before),
        "seed_landmarks_after": int(seed_landmarks_after),
        "rescue_attempted": bool(stats.get("pnp_support_rescue_attempted", stats.get("rescue_attempted", False))),
        "rescue_succeeded": bool(stats.get("pnp_support_rescue_succeeded", stats.get("rescue_succeeded", False))),
        "localisation_only_rescue_frame": bool(stats.get("localisation_only_rescue_frame", False)),
        "support_refresh_triggered": bool(stats.get("guarded_support_refresh_triggered", stats.get("support_refresh_triggered", False))),
        "local_ba_attempted": bool(stats.get("local_ba_attempted", False)),
        "local_ba_skipped": bool(stats.get("local_ba_skipped", False)),
        "local_ba_succeeded": bool(stats.get("local_ba_succeeded", False)),
        "local_ba_reason": stats.get("local_ba_reason", None),
        "local_ba_skip_reason": stats.get("local_ba_skip_reason", None),
        "local_ba_n_local_keyframes": int(stats.get("local_ba_n_local_keyframes", 0)),
        "local_ba_n_local_landmarks": int(stats.get("local_ba_n_local_landmarks", 0)),
        "local_ba_n_observations": int(stats.get("local_ba_n_observations", 0)),
        "local_ba_initial_mean_reproj_error_px": stats.get("local_ba_initial_mean_reproj_error_px", None),
        "local_ba_initial_median_reproj_error_px": stats.get("local_ba_initial_median_reproj_error_px", None),
        "local_ba_final_mean_reproj_error_px": stats.get("local_ba_final_mean_reproj_error_px", None),
        "local_ba_final_median_reproj_error_px": stats.get("local_ba_final_median_reproj_error_px", None),
        "local_ba_iterations": int(stats.get("local_ba_iterations", 0)),
        "local_ba_accepted_iterations": int(stats.get("local_ba_accepted_iterations", 0)),
        "local_ba_initial_damping": stats.get("local_ba_initial_damping", None),
        "local_ba_final_damping": stats.get("local_ba_final_damping", None),
    }


# Keep one parseable scorecard subset stable
def frame_scorecard_row(row: dict) -> dict:
    return {key: row.get(key, None) for key in STANDARD_FRAME_STAT_FIELDS}


# Format the standard per-frame console line
def format_frame_scorecard(row: dict, *, mode: str = "short") -> str:
    ref = row.get("reference_keyframe_index", None)
    ref_text = "None" if ref is None else str(int(ref))
    reason = row.get("pipeline_reason", None)
    line = (
        f"frame {int(row.get('frame_index', -1)):04d} "
        f"ref={ref_text} "
        f"ok={bool(row.get('pipeline_ok', False))} "
        f"reason={reason} "
        f"track={int(row.get('n_track_inliers', 0))} "
        f"pnp={int(row.get('n_pnp_inliers', 0))}/{int(row.get('n_pnp_corr', 0))} "
        f"new={int(row.get('n_new_added', 0))} "
        f"kf={bool(row.get('pipeline_keyframe_promoted', False))} "
        f"rescue={bool(row.get('rescue_attempted', False))}/{bool(row.get('rescue_succeeded', False))} "
        f"refresh={bool(row.get('support_refresh_triggered', False))} "
        f"landmarks={int(row.get('seed_landmarks_before', 0))}->{int(row.get('seed_landmarks_after', 0))}"
    )

    if str(mode) != "long":
        return line

    extras = []
    for key in [
        "diagnostic_n_pnp_corr",
        "diagnostic_n_pnp_inliers",
        "pnp_spatial_gate_rejected",
        "pnp_component_gate_rejected",
        "threshold_summary",
    ]:
        if key in row:
            extras.append(f"{key}={row[key]}")

    if len(extras) == 0:
        return line

    return f"{line} " + " ".join(extras)
