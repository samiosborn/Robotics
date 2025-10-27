# src.dynamics/spatial_inertia.py
from dataclasses import dataclass
import numpy as np
from src.dynamics.spatial_math import skew3

@dataclass
class PlanarLinkInertia:
   # Mass (kg)
   mass: float
   # COM in the link_frame (x, y)
   com_xy: np.ndarray
   # Planar link inertia about z-axis through COM
   Izz: float

# Spatial inertia for planar link lifted into 3D (Featherstone-style)
def spatial_inertia_6(link: PlanarLinkInertia):
   # Mass of link
   m = link.mass
   # Centre point (lifted to 3D)
   c = np.array([link.com_xy[0], link.com_xy[1], 0.0])
   # Centre in skew symmetric matrix form
   C = skew3(c)
   # Rotational inertia about CoM
   I_C = np.diag([0.0, 0.0, link.Izz])
   # Rotational inertia about origin
   I_o = I_C + m * (C @ C.T)
   # Stack
   return np.vstack([np.hstack([I_o, m*C]), np.hstack([-m*C, m*np.eye(3)])])
