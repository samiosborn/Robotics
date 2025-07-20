# forward.py
import numpy as np
from .dh import dh_transform # Same folder

# Dimensions
dim = 2

# Degrees of Freedom
DOF = 3

# Forward Kinematics
def forward_kinematics(joint_angles, joint_offsets, link_lengths, base_position):
    # Positions of joints
    positions = [base_position]

    # Keep track of a cumulative transformation matrix
    T_cum = np.eye(dim+1)

    # Apply the translation of origin
    T_cum[:dim, dim] = base_position

    # Loop through each joint
    for i in range(len(joint_angles)):

        # Define the next DH transformation
        net_joint_angle = joint_angles[i] + joint_offsets[i]
        T = dh_transform(net_joint_angle, link_lengths[i])
        
        # Update the cumulative transformation
        T_cum = T_cum @ T
        
        # Apply the cumulative transformation to the local origin
        next_position = T_cum @ np.array([0, 0, 1])

        # Append the next end joint position
        positions.append(next_position[:dim])

    # Return as numpy array
    return np.array(positions)

# FK for just end effector pose
def FK_end_effector_pose(joint_angles, joint_offsets, link_lengths, base_position):
    # Initialise the end effector pose
    end_effector_pose = np.zeros(dim+1)
    
    # Compute the FK for all joint positions
    joint_positions = forward_kinematics(joint_angles, joint_offsets, link_lengths, base_position)
    
    # Set position of end-effector 
    end_effector_pose[0:dim] = joint_positions[-1]
    
    # Calculate phi (orientation) 
    phi = np.sum(joint_angles + joint_offsets)
    end_effector_pose[2] = phi

    return joint_positions, end_effector_pose