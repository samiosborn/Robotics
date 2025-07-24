# config.py
import numpy as np

# Link Lengths (meters)
LINK_LENGTHS = np.array([0.6, 0.3, 0.1])

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