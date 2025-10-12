# demos/move_between_waypoints_task_space.py
import argparse
import numpy as np
import config
from storage.waypoints import get_waypoint
from control.trajectory_executor import execute_q_trajectory
from control.motion_controller import MotionController
from trajectory.joint_space import q_from_waypoint, plan_joint_space_quintic

# Prefer linear; use cubic/quintic task-space planners if your repo has them.
from trajectory.task_space import plan_task_space_linear_from_waypoints
try:
    from trajectory.task_space import plan_task_space_cubic_from_waypoints
except Exception:
    plan_task_space_cubic_from_waypoints = None
try:
    from trajectory.task_space import plan_task_space_quintic_from_waypoints
except Exception:
    plan_task_space_quintic_from_waypoints = None

PLANNERS = {
    "linear":  plan_task_space_linear_from_waypoints,
    "cubic":   plan_task_space_cubic_from_waypoints,
    "quintic": plan_task_space_quintic_from_waypoints,
}

# --- helpers ---
def _read_current_deg():
    ctrl = MotionController()
    try:
        return [float(ctrl.bus.read_position(j)) for j in config.JOINT_ORDER]
    finally:
        ctrl.close()

def _mean_abs_err_deg(a_deg, b_deg) -> float:
    a = np.asarray(a_deg, float)
    b = np.asarray(b_deg, float)
    return float(np.mean(np.abs(a - b)))

# --- main ---
def main():
    ap = argparse.ArgumentParser(
        description="Task-space straight-line EE move between two waypoints."
    )
    ap.add_argument("--A", required=True, help="Start waypoint name")
    ap.add_argument("--B", required=True, help="End waypoint name")
    ap.add_argument("--profile", choices=list(PLANNERS.keys()), default="linear",
                    help="Time-scaling profile (falls back to linear if unavailable).")
    ap.add_argument("--T", type=float, default=config.DEFAULT_TRAJ_DURATION, help="Total duration (s)")
    ap.add_argument("--dt", type=float, default=config.DEFAULT_TRAJ_DT, help="Sample period (s)")
    ap.add_argument("--dry-run", action="store_true", help="Plan only — do NOT move hardware")

    # Orientation control (IK is stricter with orientation ON)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--use-orientation", dest="use_orientation", action="store_true",
                     help="Track tool orientation along the path (default).")
    grp.add_argument("--no-orientation", dest="use_orientation", action="store_false",
                     help="Position-only straight line (easier IK).")
    ap.set_defaults(use_orientation=True)

    # Safety / robustness
    ap.add_argument("--go-to-start", action="store_true",
                    help="Move from CURRENT pose to waypoint A first (recommended).")
    ap.add_argument("--T-start", type=float, default=2.0,
                    help="Duration (s) for the pre-position move to A when --go-to-start is set.")
    ap.add_argument("--start-tol-deg", type=float, default=2.0,
                    help="If mean |error| to A (when A has q_deg) exceeds this, recommend pre-position.")
    args = ap.parse_args()

    wayA = get_waypoint(args.A)
    wayB = get_waypoint(args.B)

    # Optional pre-position to A using a short joint-space quintic (no external dependency).
    if args.go_to_start:
        if args.dry_run:
            print(f"[DRY] Would pre-position to '{args.A}' over {args.T_start:.2f}s before task-space move.")
        else:
            # If A has joints, use them. Otherwise, do nothing (no safe IK guess).
            if "q_deg" in wayA and wayA["q_deg"] is not None:
                qA = q_from_waypoint(wayA)  # radians
                cur_deg = _read_current_deg()
                q_cur = np.deg2rad(np.asarray(cur_deg, float))
                _, q_traj_start = plan_joint_space_quintic(q_cur, qA, total_time=args.T_start, dt=args.dt)
                execute_q_trajectory(q_traj_start, dt=args.dt)
            else:
                print(f"[WARN] Waypoint '{args.A}' lacks joint data; skipping pre-position.")
    else:
        # Warn if we're far from A when A has joint data
        if "q_deg" in wayA and wayA["q_deg"] is not None:
            try:
                cur = _read_current_deg()
                A_deg = [float(wayA["q_deg"][j]) for j in config.JOINT_ORDER]
                mae = _mean_abs_err_deg(cur, A_deg)
                if mae > args.start_tol_deg:
                    print(f"[WARN] Current ≠ start '{args.A}': mean |error| ≈ {mae:.1f}° "
                          f"(tol {args.start_tol_deg:.1f}°). Consider --go-to-start.")
            except Exception as ex:
                if config.DEBUG:
                    print(f"[WARN] Could not read current joints to compare with A: {ex}")

    # Choose planner (fallback to linear if selected one is unavailable)
    plan_fn = PLANNERS.get(args.profile)
    if plan_fn is None:
        plan_fn = plan_task_space_linear_from_waypoints
        print(f"[INFO] '{args.profile}' profile not available; using 'linear'.")

    # Plan the task-space straight line
    try:
        times, q_traj, target_poses = plan_fn(
            wayA, wayB, total_time=args.T, dt=args.dt, use_orientation=args.use_orientation
        )
    except RuntimeError as e:
        print(f"[ERROR] Task-space planning failed: {e}")
        if args.use_orientation:
            print("Hint: retry with --no-orientation (position-only) or ensure PICK/PLACE tool orientations are similar.")
        else:
            print("Hints: increase --T, reduce --dt, or pre-position with --go-to-start.")
        return

    print(f"[INFO] Planned task-space straight-line ({'with' if args.use_orientation else 'no'} orientation): "
          f"{len(times)} steps, T={times[-1]:.3f}s")

    if args.dry_run:
        print("[DRY] First pose:", target_poses[0])
        print("[DRY] Last  pose:",  target_poses[-1])
        if "q_deg" in wayA and wayA["q_deg"] is not None:
            qA = q_from_waypoint(wayA)
            print("[DRY] Start joints (deg):", dict(zip(config.JOINT_ORDER, map(float, np.rad2deg(qA)))))
        return

    execute_q_trajectory(q_traj, dt=args.dt)

if __name__ == "__main__":
    main()
