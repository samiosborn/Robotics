# src/kinematics/forward.py
import numpy as np
from src.kinematics.dh import dh_transform

# Dimensions
dim = 2

# Degrees of Freedom
DOF = 3

# Forward Kinematics
def forward_kinematics(joint_angles, joint_offsets, link_lengths, base_position):
    # Number of joints
    n = len(joint_angles)
    # Positions of joints
    joints = [base_position.copy()]
    # List of per-link 2D homogeneous transforms in (3x3) world
    T_list = []
    # Keep track of a cumulative transformation matrix
    T_cum = np.eye(dim+1)
    # Apply the translation of origin
    T_cum[:dim, dim] = base_position
    # Cumulative link angles in (3x3) world
    phis = np.zeros(n, dtype=float)
    # Initialise link angle
    phi = 0.0
    # Loop over all joints
    for i in range(n):
        # Net joint angle
        theta = joint_angles[i] + joint_offsets[i]
        # Accumulate to phi
        phi += theta
        phis[i] = phi
        # Rotate then translate in local frame
        T_local = dh_transform(theta, link_lengths[i])
        # Accumulate transformation
        T_cum = T_cum @ T_local
        T_list.append(T_cum.copy())
        # Next joint position from local origin
        next_position = T_cum @ np.array([0.0, 0.0, 1.0])
        # Append the next end joint position
        joints.append(next_position[:dim])
    # Return as numpy array
    return np.vstack(joints), phis, T_list

# FK for just end effector pose
def FK_end_effector_pose(joint_angles, joint_offsets, link_lengths, base_position):
    # Initialise the end effector pose
    end_effector_pose = np.zeros(dim+1, dtype=float)
    # Compute the FK for all joint positions
    joints, phis, _ = forward_kinematics(joint_angles, joint_offsets, link_lengths, base_position)
    # Set position of end-effector 
    end_effector_pose[0:dim] = joints[-1]
    # Calculate phi (orientation) 
    end_effector_pose[2] = phis[-1]
    return joints, end_effector_pose