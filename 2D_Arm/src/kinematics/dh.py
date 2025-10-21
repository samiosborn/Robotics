# dh.py
import numpy as np

# 2D
dim = 2

# Denavit-Hartenburg (DH) Parameter Matrix
def dh_transform(joint_angle, link_length):
    # Initialise DH matrix using identity
    T = np.eye(dim+1)

    # Define the rotation matrix segment
    cos_theta = np.cos(joint_angle)
    sin_theta = np.sin(joint_angle)
    T[0,0] = cos_theta
    T[1,0] = sin_theta
    T[0,1] = -sin_theta
    T[1,1] = cos_theta

    # Translate origin
    T[0,2] = link_length*cos_theta
    T[1,2] = link_length*sin_theta
    
    return T
