# forward.py
import numpy as np
from typing import List, Tuple
import config
from .dh import mdh
from .pose import R_to_rpy, rpy_to_R

# Build frames along kinematic chain
def fk_all_frames(q_rad: List[float]) -> List[np.ndarray]:
    # Initialise transformation matrix
    Ts = [np.eye(4)]
    for i, (a, alpha, d, theta0) in enumerate(config.MDH_PARAMS):
        # Add offset to commanded angle
        theta = theta0 + float(q_rad[i])
        # Build joint transformation
        Ti = mdh(a, alpha, d, theta)
        # Chain to previous frame
        Ts.append(Ts[-1] @ Ti)
    # Tool orientation
    roll = np.deg2rad(config.TOOL_RPY_DEG[0])
    pitch = np.deg2rad(config.TOOL_RPY_DEG[1])
    yaw = np.deg2rad(config.TOOL_RPY_DEG[2])
    # Convert tool orientation to 3x3 rotation matrix
    R_tool = rpy_to_R(roll, pitch, yaw)
    # Tool position offset
    p_tool = np.array(config.TOOL_POS_M, dtype=float).reshape(3,)
    # Single 4x4 homogeneous transform
    T_tool = np.eye(4)
    T_tool[:3,:3] = R_tool
    T_tool[:3, 3] = p_tool
    # End effector frame appended
    Ts.append(Ts[-1] @ T_tool)
    return Ts

# End effector pose as (pos xyz, rpy)
def fk_pose(q_rad: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    # Extract tool frame
    T = fk_all_frames(q_rad)[-1]
    # Position
    p = T[:3,3]
    # Rotation matrix
    R = T[:3,:3]
    # Vector (roll, pitch and yaw) fromm rotation matrix
    rpy = np.array(R_to_rpy(R))
    return p, rpy
