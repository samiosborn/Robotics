# task_space_visualisation.py
import numpy as np
from kinematics.inverse import inverse_via_DLS

# Linear Task-space Trajectory
def linear_task_space_trajectory(start_pose, end_pose, total_time, dt, joint_offsets, link_lengths, base_position):

    # Number of steps
    num_steps = int(total_time / dt) + 1
    
    # Track of joint angle positions
    joint_angles = np.zeros((num_steps, 3))

    # Current pose as initial pose
    current_pose = start_pose.copy()

    # Loop for all steps
    for i in range(num_steps):
        
        # Fraction of the path
        s = i / (num_steps - 1)

        # Current pose (straight line)
        current_pose = start_pose + s *(end_pose - start_pose)  

        # Inverse kinematics
        try:
            # Damped Least Squares solver
            current_joint_angles = inverse_via_DLS(current_pose, joint_offsets, link_lengths, base_position)
        except RuntimeError: 
            # Print failure
            print(f"IK failed at step {i} for pose {current_pose}")
            # Store NaNs
            joint_angles[i] = np.nan
            continue

        # Update tracker of joint angles
        joint_angles[i] = current_joint_angles

    # Return the joint angles
    return joint_angles