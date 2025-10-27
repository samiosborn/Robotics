# src/kinematics/forward.py
import numpy as np
from src.kinematics.dh import dh_transform
from src.dynamics.spatial_transforms import rotate3z

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
    T_cum = np.eye(3)
    # Apply the translation of origin
    T_cum[:2, 2] = base_position
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
        joints.append(next_position[:2])
    # Return as numpy array
    return np.vstack(joints), phis, T_list

# FK for just end effector pose
def FK_end_effector_pose(joint_angles, joint_offsets, link_lengths, base_position):
    # Initialise the end effector pose
    end_effector_pose = np.zeros(3, dtype=float)
    # Compute the FK for all joint positions
    joints, phis, _ = forward_kinematics(joint_angles, joint_offsets, link_lengths, base_position)
    # Set position of end-effector 
    end_effector_pose[0:2] = joints[-1]
    # Calculate phi (orientation) 
    end_effector_pose[2] = phis[-1]
    return joints, end_effector_pose

# Forward Kinematics (3D version, planar arm embedded in 3D)
def forward_kinematics_3d(joint_angles, joint_offsets, link_lengths, base_position, base_angle):
    # Number of joints
    n = len(joint_angles)
    # Positions of joints in world coordinates
    joints = [np.array(base_position, dtype=float).copy()]
    # List of per-link homogeneous transforms in 3D (4x4)
    T_list = []
    # Base rotation (about z-axis)
    R_base = rotate3z(base_angle)
    # Base transform (world to base)
    T_cum = np.eye(4)
    T_cum[:3, :3] = R_base
    T_cum[:3, 3] = np.array(base_position, dtype=float)
    # Cumulative link angle within the plane
    phis = np.zeros(n, dtype=float)
    phi = 0.0
    # Loop over all joints
    for i in range(n):
        # Net joint angle (including offsets)
        theta = joint_angles[i] + joint_offsets[i]
        # Accumulate joint angle in planar space
        phi += theta
        phis[i] = phi
        # Local planar transform (in 2D homogeneous form)
        T_local_2d = dh_transform(theta, link_lengths[i])
        # Embed 2D transform into 3D (z = 0 plane)
        T_local_3d = np.eye(4)
        T_local_3d[:2, :2] = T_local_2d[:2, :2]
        T_local_3d[:2, 3] = T_local_2d[:2, 2]
        # Accumulate total transform
        T_cum = T_cum @ T_local_3d
        T_list.append(T_cum.copy())
        # Extract joint position (3D)
        next_position = T_cum @ np.array([0.0, 0.0, 0.0, 1.0])
        joints.append(next_position[:3])
    # Return as numpy array
    return np.vstack(joints), phis, T_list

# FK for just end effector pose (3D version)
def FK_end_effector_pose_3d(joint_angles, joint_offsets, link_lengths, base_position, base_angle):
    # [x, y, z, phi]
    end_effector_pose = np.zeros(4, dtype=float) 
    # Compute FK for all joints
    joints, phis, _ = forward_kinematics_3d(joint_angles, joint_offsets, link_lengths, base_position, base_angle)
    # Set position of end-effector
    end_effector_pose[:3] = joints[-1]
    # Orientation in plane
    end_effector_pose[3] = phis[-1]
    return joints, end_effector_pose
