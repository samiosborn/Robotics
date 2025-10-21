# src/trajectory/task_space_trajectory.py
import numpy as np
from src.kinematics.inverse import inverse_via_DLS

# Linear Task-space Trajectory
def linear_task_space_trajectory(start_pose, end_pose, total_time, dt, joint_offsets, link_lengths, base_position):

   # Number of steps
    num_steps = int(total_time / dt) + 1
    
    # Time vector
    times = np.linspace(0, total_time, num_steps)

    # Track of joint angle values
    joint_angles = np.zeros((num_steps, 3))

    # Delta joint angles
    delta_pose = end_pose - start_pose

    # Double check start and end poses are reachable first
    if inverse_via_DLS(start_pose, joint_offsets, link_lengths, base_position) is None or inverse_via_DLS(end_pose, joint_offsets, link_lengths, base_position) is None:
        raise ValueError("IK failed for start or end pose – cannot plan trajectory")

    # Loop for all steps
    for i, t in enumerate(times):
        
        # Fraction of the path
        s = t / total_time

        # Current pose (straight line)
        current_pose = start_pose + s * delta_pose 

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
    return times, joint_angles
