# control/trajectory_executor.py
import time
import numpy as np
import config
from control.motion_controller import MotionController
from trajectory.joint_space import plan_joint_space_quintic  # use same planner for soft start

# --- helpers ---------------------------------------------------------------

def _deg_limits_check(q_deg_row: np.ndarray) -> None:
    for j_name, q in zip(config.JOINT_ORDER, q_deg_row):
        lo, hi = config.JOINT_LIMITS_DEG[j_name]
        if not (lo <= q <= hi):
            raise ValueError(f"{j_name} command {q:.1f}° out of limits [{lo:.1f},{hi:.1f}]°")

def _write_joint_row(ctrl: MotionController, q_deg_row: np.ndarray) -> bool:
    moved_all = True
    for j_name, q in zip(config.JOINT_ORDER, q_deg_row):
        moved_all &= ctrl.bus.write_position(j_name, float(q))
    return moved_all

def _read_current_q_deg(ctrl: MotionController) -> np.ndarray:
    # Best-effort readback — adjust if your bus exposes different API names
    qs = []
    for j_name in config.JOINT_ORDER:
        try:
            # Expect a float in degrees
            q = ctrl.bus.read_position(j_name)
        except AttributeError:
            # Fallback: try <bus>.read_<name> or raise cleanly
            raise RuntimeError("Motion bus does not expose read_position(joint). "
                               "Please implement it or provide a current-q hook.")
        qs.append(float(q))
    return np.array(qs, dtype=float)

# --- public API ------------------------------------------------------------

def execute_q_trajectory(q_traj_rad: np.ndarray, dt: float = config.DEFAULT_TRAJ_DT) -> None:
    """
    Streams positions at fixed dt. Enforces joint limits (deg) before write.
    """
    q_traj_deg = np.rad2deg(q_traj_rad).astype(float)
    ctrl = MotionController()
    try:
        for q_deg in q_traj_deg:
            _deg_limits_check(q_deg)
            moved_all = _write_joint_row(ctrl, q_deg)
            if config.DEBUG and not moved_all:
                print("[WARN] One or more joints rejected a setpoint this tick.")
            time.sleep(dt)
    finally:
        ctrl.close()

def execute_q_trajectory_soft_start(
    q_traj_rad: np.ndarray,
    dt: float = config.DEFAULT_TRAJ_DT,
    pre_T: float = 1.5,
    start_tol_deg: float = 0.5,
) -> None:
    """
    Like execute_q_trajectory, but first blends from CURRENT encoder pose to the
    first waypoint with a short quintic ramp so it doesn't lunge to the start.
    """
    q_traj_deg = np.rad2deg(q_traj_rad).astype(float)
    if q_traj_deg.ndim != 2 or q_traj_deg.shape[1] != len(config.JOINT_ORDER):
        raise ValueError("q_traj_rad must be [N x n_joints]")

    ctrl = MotionController()
    try:
        q_now_deg = _read_current_q_deg(ctrl)
        q_start_deg = q_traj_deg[0].copy()

        # If we're already close to start, skip the pre-ramp
        if np.max(np.abs(q_now_deg - q_start_deg)) > start_tol_deg and pre_T > 0.0:
            q_now = np.deg2rad(q_now_deg)
            q_start = np.deg2rad(q_start_deg)
            # Short, minimum-jerk join
            _, q_pre = plan_joint_space_quintic(q_now, q_start, total_time=pre_T, dt=dt)
            if config.DEBUG:
                print(f"[INFO] Soft-start: planning current→start in {pre_T:.2f}s "
                      f"({len(q_pre)} steps, max Δ={np.max(np.abs(q_now_deg-q_start_deg)):.1f}°)")
            # Stream the join
            for q_deg in np.rad2deg(q_pre):
                _deg_limits_check(q_deg)
                moved_all = _write_joint_row(ctrl, q_deg)
                if config.DEBUG and not moved_all:
                    print("[WARN] One or more joints rejected a setpoint (soft-start).")
                time.sleep(dt)
        else:
            if config.DEBUG:
                print("[INFO] Soft-start skipped: already near start.")

        # Now stream the planned trajectory
        for q_deg in q_traj_deg:
            _deg_limits_check(q_deg)
            moved_all = _write_joint_row(ctrl, q_deg)
            if config.DEBUG and not moved_all:
                print("[WARN] One or more joints rejected a setpoint this tick.")
            time.sleep(dt)
    finally:
        ctrl.close()

def move_to_waypoint(
    q_target_rad: np.ndarray,
    T: float = 2.0,
    dt: float = config.DEFAULT_TRAJ_DT,
) -> None:
    """
    One-shot: read current encoders, plan a quintic to q_target, stream it.
    Useful for “just go there smoothly”.
    """
    ctrl = MotionController()
    try:
        q_now_deg = _read_current_q_deg(ctrl)
        q_now = np.deg2rad(q_now_deg)
        q_tgt = np.array(q_target_rad, dtype=float).reshape(-1)
        _, q_pre = plan_joint_space_quintic(q_now, q_tgt, total_time=T, dt=dt)
        if config.DEBUG:
            print(f"[INFO] Move-to-waypoint: {len(q_pre)} steps over {T:.2f}s")
        for q_deg in np.rad2deg(q_pre):
            _deg_limits_check(q_deg)
            moved_all = _write_joint_row(ctrl, q_deg)
            if config.DEBUG and not moved_all:
                print("[WARN] One or more joints rejected a setpoint (move_to_waypoint).")
            time.sleep(dt)
    finally:
        ctrl.close()
