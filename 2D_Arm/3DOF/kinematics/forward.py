# forward.py
import numpy as np
from .dh import dh_transform # Same folder

# Dimensions
dim = 2

# Forward Kinematics
def forward_kinematics(joint_angles, INITIAL_JOINT_ANGLES, LINK_LENGTHS, STARTING_POSITION):
    # Positions of joints
    positions = [STARTING_POSITION]

    # Keep track of a cumulative transformation matrix
    T_cum = np.eye(dim + 1)

    # Apply any translation of origin
    T_cum[:2, 2] = STARTING_POSITION

    # Loop through each joint
    for i in range(len(joint_angles)):

        # Define the next DH transformation
        net_joint_angle = joint_angles[i] + INITIAL_JOINT_ANGLES[i]
        T = dh_transform(net_joint_angle, LINK_LENGTHS[i])
        
        # Update the cumulative transformation
        T_cum = T_cum @ T
        
        # Apply the cumulative transformation to the local origin
        next_position = T_cum @ np.array([0, 0, 1])

        # Append the next end joint position
        positions.append(next_position[:2])

    # Return as numpy array
    return np.array(positions)
