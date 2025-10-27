# config.py
import numpy as np

# Gravity
g = 9.81

# Link Lengths (meters)
LINK_LENGTHS = np.array([0.6, 0.3, 0.1])

# Link masses (kg)
LINK_MASSES = np.array([1.2, 0.6, 0.2])

# Center of Mass (CoM)
LINK_COM_LOCAL = np.stack([
    np.array([LINK_LENGTHS[0]/2, 0.0]),
    np.array([LINK_LENGTHS[1]/2, 0.0]),
    np.array([LINK_LENGTHS[2]/2, 0.0])
])

# Inertia about z axis (through CoM)
LINK_Izz_COM = (1.0 / 12.0) * LINK_MASSES * (LINK_LENGTHS**2)

# Joint Offsets (radians)
JOINT_OFFSETS = np.array([0, 0, 0])

# Base Position (meters)
BASE_POSITION = np.array([0, 0])

# FK Time Delta (seconds)
FK_TIME_DELTA = 0.01

# Trajectory Time (seconds)
TRAJECTORY_TIME = 5

# Animation Time (seconds)
ANIMATION_TIME = 3