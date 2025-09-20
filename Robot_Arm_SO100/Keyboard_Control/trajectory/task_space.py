# trajectory/task_space.py
import numpy as np
from typing import List, Tuple, Optional
import config
from kinematics.inverse import inverse_via_DLS, inverse_via_DLS_position_only
from kinematics.forward import fk_pose
from kinematics.pose import rpy_to_R, R_to_rpy, hat, vee, so3_log, so3_exp

# Linear interpolate pose on SE(3) between (p0, R0) to (p1, R1)
def linear_interp(p0: np.ndarray, R0: np.ndarray,
                p1: np.ndarray, R1: np.ndarray,
                s: float) -> Tuple[np.ndarray, np.ndarray]:
    # Linear position movement: p(s) = p0 + s * (p1 - p0)
    p = (1.0 - s) * p0 + s * p1
    # Relative rotation
    R_rel = R0.T @ R1
    # Convert rotation matrix to rotation vector
    w = so3_log(R_rel)
    # Linear rotation: R(s) = R0 * exp( s * log(R0^T R1) )
    R = R0 @ so3_exp(s * w)
    return p, R

# Linear interpolate position between (p0) to (p1)
def linear_interp_position_only(p0: np.ndarray, p1: np.ndarray, 
                  s: float) -> Tuple[np.ndarray, np.ndarray]:
    # Linear position movement: p(s) = p0 + s * (p1 - p0)
    p = (1.0 - s) * p0 + s * p1
    # Return position only
    return p

# Build a sequence of target poses along a straight line in SE(3)
def build_linear_pose_profile(start_pose_rpy: np.ndarray,
                              end_pose_rpy: np.ndarray,
                              total_time: float,
                              dt: float,
                              use_orientation: bool = True) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    # Unpack start and end
    p0 = np.array(start_pose_rpy[:3], dtype=float)
    rpy0 = np.array(start_pose_rpy[3:], dtype=float)
    p1 = np.array(end_pose_rpy[:3], dtype=float)
    rpy1 = np.array(end_pose_rpy[3:], dtype=float)

    # Convert RPY to rotation matrices
    R0 = rpy_to_R(rpy0[0], rpy0[1], rpy0[2])
    R1 = rpy_to_R(rpy1[0], rpy1[1], rpy1[2])

    # Time vector
    n_steps = int(np.round(total_time / dt)) + 1
    times = np.linspace(0.0, total_time, n_steps)

    # Build list of poses as (pos_xyz, rpy) tuples
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, t in enumerate(times):
        # s in [0.0, 1.0]
        s = 0.0 if total_time <= 0.0 else t / total_time
        s = float(np.clip(s, 0.0, 1.0))
        # Linear interpolation
        if use_orientation:
            p, R = linear_interp(p0, R0, p1, R1, s)
        else:
            p = linear_interp_position_only(p0, p1, s)
            R = R0
        # Convert orientation to RPY
        rpy = np.array(R_to_rpy(R))
        poses.append((p, rpy))
    return times, poses

# Plan a task-space linear trajectory
def plan_task_space_linear(start_pose_rpy: np.ndarray,
                           end_pose_rpy: np.ndarray,
                           total_time: float = config.DEFAULT_TRAJ_DURATION,
                           dt: float = config.DEFAULT_TRAJ_DT,
                           q_seed_rad: Optional[List[float]] = None,
                           use_orientation: bool = True,
                           rot_weight: float = config.IK_ROT_WEIGHT,
                           lambda_damp: float = config.IK_DAMPING_LAMBDA,
                           max_iters: int = config.IK_MAX_ITERS) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    # Make list of target poses
    times, target_poses = build_linear_pose_profile(
        np.array(start_pose_rpy, dtype=float),
        np.array(end_pose_rpy, dtype=float),
        total_time,
        dt,
        use_orientation=use_orientation
    )

    # Number of steps
    n = len(times)
    # Allocate joint trajectory
    q_traj = np.zeros((n, config.DOF), dtype=float)

    # Seed of q
    if q_seed_rad is None:
        q_seed = np.zeros(config.DOF, dtype=float)
    else:
        q_seed = np.array(q_seed_rad, dtype=float)

    # Step through each target pose
    for k, (p_tgt, rpy_tgt) in enumerate(target_poses):
        if use_orientation:
            pose = np.hstack([p_tgt, rpy_tgt])
            # Solve IK
            q_sol = inverse_via_DLS(
                pose,
                theta_initial=q_seed,
                rot_weight=rot_weight,
                lambda_damp=lambda_damp,
                max_iterations=max_iters,
                pos_tol=config.IK_POS_TOL_M,
                ang_tol=config.IK_ANG_TOL_RAD
            )
        else:
            # Solve IK (position-only)
            q_sol = inverse_via_DLS_position_only(
                p_tgt,
                theta_initial=q_seed,
                lambda_damp=lambda_damp,
                max_iterations=max_iters,
                pos_tol=config.IK_POS_TOL_M
            )

        # Store and roll the seed
        q_traj[k, :] = q_sol
        q_seed = q_sol

    return times, q_traj, target_poses

# Extract a pose [x,y,z, r,p,y] from a waypoint (dict of joint names and joint angles)
def pose_from_waypoint(w: dict) -> np.ndarray:
    # If waypoint has "pose", use it
    if "pose" in w and w["pose"] is not None:
        return np.array(w["pose"], dtype=float)
    # Else, compute pose using FK
    q_deg_dict = w["q_deg"]
    q_vec_deg = [q_deg_dict[j] for j in config.JOINT_ORDER]
    q_rad = np.deg2rad(q_vec_deg)
    p, rpy = fk_pose(q_rad)
    return np.hstack([p, rpy])

# Plan linear trajectory between two waypoints 
def plan_task_space_linear_from_waypoints(wayA: dict,
                                          wayB: dict,
                                          total_time: float = config.DEFAULT_TRAJ_DURATION,
                                          dt: float = config.DEFAULT_TRAJ_DT,
                                          use_orientation: bool = True,
                                          rot_weight: float = config.IK_ROT_WEIGHT,
                                          lambda_damp: float = config.IK_DAMPING_LAMBDA,
                                          max_iters: int = config.IK_MAX_ITERS) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    # Extract pose from waypoints
    poseA = pose_from_waypoint(wayA)
    poseB = pose_from_waypoint(wayB)

    # Use waypoint A's joint angles as the initial IK seed if available
    if "q_deg" in wayA and wayA["q_deg"] is not None:
        qA_deg = [wayA["q_deg"][j] for j in config.JOINT_ORDER]
        q_seed = np.deg2rad(qA_deg)
    else:
        q_seed = None

    # Execute the linear task space planner between the start and end pose
    return plan_task_space_linear(
        start_pose=poseA,
        end_pose=poseB,
        total_time=total_time,
        dt=dt,
        q_seed_rad=q_seed,
        use_orientation=use_orientation,
        rot_weight=rot_weight,
        lambda_damp=lambda_damp,
        max_iters=max_iters
    )
