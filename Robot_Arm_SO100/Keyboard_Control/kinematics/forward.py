# kinematics/forward.py
import numpy as np
from typing import List, Tuple, Dict, Iterable
import config
from .dh import mdh, mdh_fixed
from .pose import R_to_rpy, rpy_to_R

def _group_fixed_inserts(inserts: Iterable[tuple]) -> Dict[int, list]:
    """
    Group MDH_FIXED_INSERTS by target index (before joint with that index),
    but do not change order within each group.
    Each group will be composed into ONE transform to avoid bumping frame count.
    """
    grouped: Dict[int, list] = {}
    for idx, spec in inserts:
        grouped.setdefault(int(idx), []).append(spec)
    return grouped

def _compose_fixed(specs: List[tuple]) -> np.ndarray:
    """
    Compose a list of fixed insert specs into a single 4x4 transform.
    Each spec is one of:
      ("tx", float), ("ty", float), ("tz", float), ("mat", 4x4-like)
    """
    T = np.eye(4, dtype=float)
    for s in specs:
        kind = s[0].lower()
        if kind == "mat":
            T = T @ mdh_fixed("mat", s[1])
        else:
            T = T @ mdh_fixed(kind, s[1])
    return T

# Build frames along kinematic chain
def fk_all_frames(q_rad: List[float]) -> List[np.ndarray]:
    # Preprocess fixed inserts (e.g., shoulder Tx/Tz) so each index yields ONE composite T
    grouped = _group_fixed_inserts(getattr(config, "MDH_FIXED_INSERTS", []))
    composed: Dict[int, np.ndarray] = {k: _compose_fixed(v) for k, v in grouped.items()}

    Ts = [np.eye(4, dtype=float)]  # base frame

    # MDH joint chain
    for i, (a, alpha, d, theta0) in enumerate(config.MDH_PARAMS):
        theta = theta0 + float(q_rad[i])
        Ti = mdh(a, alpha, d, theta)
        Ts.append(Ts[-1] @ Ti)

        # If there is a fixed insert BEFORE the next joint (i+1), add it as ONE frame
        if (i + 1) in composed:
            Ts.append(Ts[-1] @ composed[i + 1])

    # Tool transform (RPY + xyz)
    roll, pitch, yaw = np.deg2rad(config.TOOL_RPY_DEG).astype(float)
    R_tool = rpy_to_R(roll, pitch, yaw)
    p_tool = np.array(config.TOOL_POS_M, dtype=float).reshape(3,)

    T_tool = np.eye(4, dtype=float)
    T_tool[:3, :3] = R_tool
    T_tool[:3, 3] = p_tool

    Ts.append(Ts[-1] @ T_tool)
    return Ts

# End effector pose as (pos xyz, rpy)
def fk_pose(q_rad: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    T = fk_all_frames(q_rad)[-1]
    p = T[:3, 3]
    R = T[:3, :3]
    rpy = np.array(R_to_rpy(R))
    return p, rpy
