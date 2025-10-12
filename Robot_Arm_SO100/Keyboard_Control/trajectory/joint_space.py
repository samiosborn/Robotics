# trajectory/joint_space.py
import numpy as np
from typing import Tuple, Dict
import config
from kinematics.inverse import inverse_via_DLS

# Ensure duration respects velocity limits
def _ensure_duration_respects_velocity_limits(q0_rad, q1_rad, total_time, profile: str) -> float:
    dq_deg = np.abs(np.rad2deg(q1_rad - q0_rad))
    vmax_vec = np.array([config.MAX_DEG_PER_SEC[j] for j in config.JOINT_ORDER], dtype=float)

    # Peak factor k based on time-scaling profile
    if profile == "linear":
        k = 1.0
    elif profile == "cubic":
        k = 1.5
    elif profile == "quintic":
        k = 1.875
    else:
        k = 1.0

    # Calculate the required minimum time (at max speed)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_joint_T = k * np.where(vmax_vec > 0, dq_deg / vmax_vec, 0.0)
        required_T = float(np.max(per_joint_T))

    # New total time
    T_new = max(total_time, required_T, 1e-9)
    if config.DEBUG and T_new > total_time + 1e-9:
        print(f"[DEBUG] Trajectory duration stretched {total_time:.3f}s to {T_new:.3f}s to respect joint velocity limits (k={k}).")
    return T_new

# Extract q from waypoint dict
def q_from_waypoint(way: Dict, q_seed_rad: np.ndarray | None = None) -> np.ndarray:
    # Take from q_deg if available
    if "q_deg" in way and way["q_deg"] is not None:
        q_deg = [way["q_deg"][j] for j in config.JOINT_ORDER]
        return np.deg2rad(q_deg)

    # Else calculate q_deg from pose (if available)
    if "pose" in way and way["pose"] is not None:
        pose = np.array(way["pose"], dtype=float).reshape(-1)
        if pose.size != 6:
            raise ValueError("Waypoint 'pose' must be length 6: [x, y, z, r, p, y].")
        q0 = np.zeros(config.DOF) if q_seed_rad is None else np.array(q_seed_rad, dtype=float)
        return inverse_via_DLS(
            pose,
            theta_initial=q0,
            rot_weight=config.IK_ROT_WEIGHT,
            lambda_damp=config.IK_DAMPING_LAMBDA,
            max_iterations=config.IK_MAX_ITERS,
            pos_tol=config.IK_POS_TOL_M,
            ang_tol=config.IK_ANG_TOL_RAD
        )

    raise ValueError("Waypoint must contain 'q_deg' or 'pose'.")

# --- TIME SCALING ---
# Linear time profile
def _linear_profile(t: float, T: float) -> float:
    # t is the current time, T is the total duration
    if T <= 0.0:
        # Avoid t/T singularity
        return 1.0
    u = float(np.clip(t / T, 0.0, 1.0))
    return u
# Cubic time profile
def _cubic_profile(t: float, T: float) -> float:
    if T <= 0.0:
        return 1.0
    u = np.clip(t / T, 0.0, 1.0)
    return float(3.0 * u**2 - 2.0 * u**3)
# Quintic time profile
def _quintic_profile(t: float, T: float) -> float:
    if T <= 0.0:
        return 1.0
    u = np.clip(t / T, 0.0, 1.0)
    return float(10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5)

# Helper: Profile name from time-scaling function
def _profile_label(f) -> str:
    if f is _linear_profile: return "linear"
    if f is _cubic_profile: return "cubic"
    if f is _quintic_profile: return "quintic"
    return "linear"

# Create trajectory profile in joint space
def _plan_with_profile(q_start_rad, q_end_rad, total_time: float, dt: float, time_scaling_fun) -> Tuple[np.ndarray, np.ndarray]:

    # Cast and flatten
    q0 = np.array(q_start_rad, dtype=float).reshape(-1)
    q1 = np.array(q_end_rad, dtype=float).reshape(-1)

    # Ensure duration meets velocity limits
    total_time = _ensure_duration_respects_velocity_limits(q0, q1, total_time, _profile_label(time_scaling_fun))

    # Hard-check limits at endpoints
    for name, q_deg in zip(config.JOINT_ORDER, np.rad2deg(q0)):
        lo, hi = config.JOINT_LIMITS_DEG[name]
        if not (lo <= q_deg <= hi):
            raise ValueError(f"Start joint {name}={q_deg:.2f}° outside limits [{lo:.1f},{hi:.1f}]°")
    for name, q_deg in zip(config.JOINT_ORDER, np.rad2deg(q1)):
        lo, hi = config.JOINT_LIMITS_DEG[name]
        if not (lo <= q_deg <= hi):
            raise ValueError(f"End joint {name}={q_deg:.2f}° outside limits [{lo:.1f},{hi:.1f}]°")

    # Number of steps
    n_steps = max(1, int(np.round(total_time / dt)) + 1)
    # Time vector
    times = np.linspace(0.0, max(total_time, 0.0), n_steps)
    # Allocate interpolated q
    q_traj = np.zeros((n_steps, config.DOF), dtype=float)
    # Joint angle delta between start and end
    dq = q1 - q0

    for i, t in enumerate(times):
        # Scale time accordingly
        s = time_scaling_fun(t, total_time)
        # Interpolate
        q_traj[i, :] = q0 + s * dq

    return times, q_traj

# --- TRAJECTORY PLANNERS FROM Q ---
# Linear joint-space planner from q
def plan_joint_space_linear(q_start_rad,
                            q_end_rad,
                            total_time: float = config.DEFAULT_TRAJ_DURATION,
                            dt: float = config.DEFAULT_TRAJ_DT) -> Tuple[np.ndarray, np.ndarray]:
    return _plan_with_profile(q_start_rad, q_end_rad, total_time, dt, _linear_profile)
# Cubic joint-space planner from q
def plan_joint_space_cubic(q_start_rad,
                           q_end_rad,
                           total_time: float = config.DEFAULT_TRAJ_DURATION,
                           dt: float = config.DEFAULT_TRAJ_DT) -> Tuple[np.ndarray, np.ndarray]:
    return _plan_with_profile(q_start_rad, q_end_rad, total_time, dt, _cubic_profile)
# Quintic joint-space planner from q
def plan_joint_space_quintic(q_start_rad,
                             q_end_rad,
                             total_time: float = config.DEFAULT_TRAJ_DURATION,
                             dt: float = config.DEFAULT_TRAJ_DT) -> Tuple[np.ndarray, np.ndarray]:
    return _plan_with_profile(q_start_rad, q_end_rad, total_time, dt, _quintic_profile)

# --- TRAJECTORY PLANNERS FROM WAYPOINT ---
# Linear joint-space planner from waypoints
def plan_joint_space_linear_from_waypoints(wayA, wayB, total_time=config.DEFAULT_TRAJ_DURATION, dt=config.DEFAULT_TRAJ_DT):
    qA_seed = None
    if "q_deg" in wayA and wayA["q_deg"] is not None:
        qA_seed = np.deg2rad([wayA["q_deg"][j] for j in config.JOINT_ORDER])

    qA = q_from_waypoint(wayA, q_seed_rad=qA_seed)
    qB = q_from_waypoint(wayB, q_seed_rad=qA)

    return plan_joint_space_linear(qA, qB, total_time=total_time, dt=dt)
# Cubic joint-space planner from waypoints
def plan_joint_space_cubic_from_waypoints(wayA, wayB, total_time=config.DEFAULT_TRAJ_DURATION, dt=config.DEFAULT_TRAJ_DT):
    qA_seed = None
    if "q_deg" in wayA and wayA["q_deg"] is not None:
        qA_seed = np.deg2rad([wayA["q_deg"][j] for j in config.JOINT_ORDER])

    qA = q_from_waypoint(wayA, q_seed_rad=qA_seed)
    qB = q_from_waypoint(wayB, q_seed_rad=qA)

    return plan_joint_space_cubic(qA, qB, total_time=total_time, dt=dt)
# Quintic joint-space planner from waypoints
def plan_joint_space_quintic_from_waypoints(wayA, wayB, total_time=config.DEFAULT_TRAJ_DURATION, dt=config.DEFAULT_TRAJ_DT):
    qA_seed = None
    if "q_deg" in wayA and wayA["q_deg"] is not None:
        qA_seed = np.deg2rad([wayA["q_deg"][j] for j in config.JOINT_ORDER])

    qA = q_from_waypoint(wayA, q_seed_rad=qA_seed)
    qB = q_from_waypoint(wayB, q_seed_rad=qA)

    return plan_joint_space_quintic(qA, qB, total_time=total_time, dt=dt)
