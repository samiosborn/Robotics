# demos/move_between_waypoints_joint_space.py
import argparse
from typing import Literal
import numpy as np
import config

from storage.waypoints import get_waypoint
from trajectory.joint_space import (
    q_from_waypoint,
    plan_joint_space_linear,
    plan_joint_space_cubic,
    plan_joint_space_quintic,
)
from control.trajectory_executor import execute_q_trajectory
from control.motion_controller import MotionController

PLANNERS = {
    "linear":  plan_joint_space_linear,
    "cubic":   plan_joint_space_cubic,
    "quintic": plan_joint_space_quintic,
}

def _read_current_deg() -> list:
    ctrl = MotionController()
    try:
        return [float(ctrl.bus.read_position(j)) for j in config.JOINT_ORDER]
    finally:
        ctrl.close()

def _read_current_rad() -> np.ndarray:
    return np.deg2rad(np.asarray(_read_current_deg(), dtype=float))

def _mean_abs_err_deg(a_deg, b_deg) -> float:
    a = np.asarray(a_deg, float)
    b = np.asarray(b_deg, float)
    return float(np.mean(np.abs(a - b)))

def _preposition_to_A(qA: np.ndarray, T: float, dt: float, profile: str, dry: bool):
    try:
        q_cur = _read_current_rad()
    except Exception as ex:
        print(f"[WARN] Could not read current joints for pre-position: {ex}")
        return
    plan_fn = PLANNERS[profile]
    _, q_pre = plan_fn(q_cur, qA, total_time=T, dt=dt)
    if dry:
        print(f"[DRY] Pre-position would move from current → A over {T:.2f}s ({profile}).")
        return
    execute_q_trajectory(q_pre, dt=dt)

def main():
    ap = argparse.ArgumentParser(description="Joint-space trajectory between two waypoints.")
    ap.add_argument("--A", required=True, help="Start waypoint name")
    ap.add_argument("--B", required=True, help="End waypoint name")
    ap.add_argument("--profile", choices=list(PLANNERS), default=config.DEFAULT_TRAJ_KIND)
    ap.add_argument("--T", type=float, default=config.DEFAULT_TRAJ_DURATION, help="Total duration (s)")
    ap.add_argument("--dt", type=float, default=config.DEFAULT_TRAJ_DT, help="Sample period (s)")
    ap.add_argument("--dry-run", action="store_true", help="Plan only — do NOT move hardware")

    # Pre-positioning (modern flags)
    ap.add_argument("--go-to-start", action="store_true",
                    help="FORCE a move from CURRENT pose to waypoint A before main move.")
    ap.add_argument("--T-start", type=float, default=2.0,
                    help="Duration (s) for the pre-position move to A when --go-to-start is set.")
    ap.add_argument("--start-tol-deg", type=float, default=2.0,
                    help="Warn if mean |error| to A exceeds this when not pre-positioning.")

    # Back-compat / conditional “soft-start” flags (aliases)
    ap.add_argument("--soft-start", action="store_true",
                    help="If far from A, do a brief pre-position to A before main move.")
    ap.add_argument("--soft-start-T", dest="soft_start_T", type=float, default=None,
                    help="Duration (s) for soft-start pre-position (defaults to --T-start).")
    ap.add_argument("--soft-start-tol-deg", dest="soft_start_tol_deg", type=float, default=None,
                    help="Tolerance (deg) for deciding to soft-start (defaults to --start-tol-deg).")

    args = ap.parse_args()

    wayA = get_waypoint(args.A)
    wayB = get_waypoint(args.B)
    qA = q_from_waypoint(wayA)  # radians
    qB = q_from_waypoint(wayB)  # radians

    # Decide on pre-positioning behavior
    do_soft_start = args.soft_start or (args.soft_start_T is not None or args.soft_start_tol_deg is not None)
    soft_T = args.soft_start_T if args.soft_start_T is not None else args.T_start
    soft_tol = args.soft_start_tol_deg if args.soft_start_tol_deg is not None else args.start_tol_deg

    if args.go_to_start:
        if args.dry_run:
            print(f"[DRY] Would pre-position to '{args.A}' over {args.T_start:.2f}s before main move.")
        _preposition_to_A(qA, T=args.T_start, dt=args.dt, profile=args.profile, dry=args.dry_run)

    elif do_soft_start:
        # Only pre-position if we're far from A
        try:
            cur_deg = _read_current_deg()
            A_deg = list(np.rad2deg(qA))
            mae = _mean_abs_err_deg(cur_deg, A_deg)
            if mae > soft_tol:
                print(f"[INFO] Soft-start: mean |error| to A ≈ {mae:.2f}° > {soft_tol:.2f}° → pre-positioning.")
                _preposition_to_A(qA, T=soft_T, dt=args.dt, profile="quintic", dry=args.dry_run)
            else:
                print(f"[INFO] Soft-start: error {mae:.2f}° ≤ {soft_tol:.2f}° → skipping pre-position.")
        except Exception as ex:
            if config.DEBUG:
                print(f"[WARN] Could not read current joints for soft-start: {ex}")

    else:
        # Just warn if we're far from A
        try:
            cur_deg = _read_current_deg()
            A_deg = list(np.rad2deg(qA))
            mae = _mean_abs_err_deg(cur_deg, A_deg)
            if mae > args.start_tol_deg:
                print(f"[WARN] Current ≠ start '{args.A}': mean |error| ≈ {mae:.1f}° "
                      f"(tol {args.start_tol_deg:.1f}°). Consider --go-to-start or --soft-start.")
        except Exception as ex:
            if config.DEBUG:
                print(f"[WARN] Could not read current joints to compare with A: {ex}")

    # Plan main trajectory A -> B
    plan_fn = PLANNERS[args.profile]
    times, q_traj = plan_fn(qA, qB, total_time=args.T, dt=args.dt)

    if config.DEBUG:
        print(f"[INFO] Planned {args.profile} joint-space trajectory: {len(times)} steps, "
              f"T={times[-1]:.3f}s")
    if args.dry_run:
        q0_deg = np.rad2deg(qA); q1_deg = np.rad2deg(qB)
        if config.DEBUG:
            print("[DRY] Start deg:", dict(zip(config.JOINT_ORDER, map(float, q0_deg))))
            print("[DRY] End   deg:", dict(zip(config.JOINT_ORDER, map(float, q1_deg))))
        return

    execute_q_trajectory(q_traj, dt=args.dt)

if __name__ == "__main__":
    main()
