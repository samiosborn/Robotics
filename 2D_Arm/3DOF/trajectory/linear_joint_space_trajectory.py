# linear_joint_space_trajectory.py
import numpy as np
from kinematics.inverse import inverse_via_DLS

# Piecewise Linear Joint-space Trajectory
def piecewise_linear_joint_space_trajectory(start_pose, end_pose, total_time, dt, joint_offsets, link_lengths, base_position):

    # Number of steps
    num_steps = int(total_time / dt) + 1
    
    # Track of joint angle values
    joint_angles = np.zeros((num_steps, 3))

    # Start joint angles
    start_joint_angles = inverse_via_DLS(start_pose, joint_offsets, link_lengths, base_position)

    # End joint angles
    end_joint_angles = inverse_via_DLS(end_pose, joint_offsets, link_lengths, base_position)

    # Raise error if IK fails
    if start_joint_angles is None or end_joint_angles is None:
        raise ValueError("IK failed for start or end pose - trajectory cannot be generated")

    # Current joint angles as initial joint angles
    current_joint_angles = start_joint_angles

    # Loop for all steps
    for i in range(num_steps):
        
        # Fraction of the path
        s = i / (num_steps - 1)

        # Current joint angles - linear interpolation
        current_joint_angles = start_joint_angles + s *(end_joint_angles - start_joint_angles) 

        # Update tracker of joint angles
        joint_angles[i] = current_joint_angles

    # Return the joint angles
    return joint_angles