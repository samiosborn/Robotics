# notebooks/examples/01_lie_se3.py
import numpy as np

# Q1 - Coordinate Transforms
# Create the rotation between coordinate frames W to C
# Set as a 3x3 rotation matrix R_{CW} that rotates 30 degrees around the y-axis
rotation_CW = np.array([
    [np.cos(np.pi/6), 0, np.sin(np.pi/6)],
    [0, 1, 0],
    [-np.sin(np.pi/6), 0, np.cos(np.pi/6)]]
    )
# Define the translation vector between coordinate frames t_{CW} = [0.5, 0.0, 1.0]^T
translate_CW = np.array([0.5, 0.0, 1.0])
print("Translate from world to camera: ", translate_CW)
# Pick a world point p_W = [1.0, 0.5, 2.0]^T
point_W = np.array([1.0, 0.5, 2.0])
print("Point in world frame: ", point_W)
# What is the coordinates of p_W in the camera frame? 
point_C = rotation_CW @ (point_W - translate_CW)
print("Point in camera frame: ", point_C)
# Verify this through an inverse operation
# Invert the rotation matrix
rotation_WC = rotation_CW.T
# Project onto world frame
point_W_est = (rotation_WC @ point_C) + translate_CW
print("Estimate of point back in world frame: ", point_W_est)

