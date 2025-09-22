# kinematics/jacobian.py
import numpy as np
from typing import List
from .forward import fk_all_frames

# Geometric Jacobian matrix: J = [ Jv ; Jr ]
def geometric_jacobian(q_rad: List[float]) -> np.ndarray:
    # Compute all frames along the kinematic chain
    Ts = fk_all_frames(q_rad)
    # Position of end effector (in base coordinates)
    p_e = Ts[-1][:3,3]
    # Initialise linear component
    Jv = np.zeros((3,6), dtype=float)
    # Initialise angular component
    Jr = np.zeros((3,6), dtype=float)
    for i in range(1, 7):
        # Transformation of frame {i-1} (in base frame)
        Ti_1 = Ts[i-1]
        # Extract Z-axis (in base coordinates)
        z = Ti_1[:3,2]
        # Position of origin 
        p = Ti_1[:3,3]
        # Linear velocity part: Jv_i = z_(i-1) x (p_e - p_(i-1))
        Jv[:, i-1] = np.cross(z, (p_e - p))
        # Revolute portion: Jr_i = z_(i-1)
        Jr[:, i-1] = z
    return np.vstack([Jv, Jr])
