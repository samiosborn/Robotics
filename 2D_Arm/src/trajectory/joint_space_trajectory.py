# src/trajectory/joint_space_trajectory.py
import numpy as np
from src.kinematics.inverse import inverse_via_DLS

# Linear Joint-space Trajectory
def linear_joint_space_trajectory(start_pose, end_pose, total_time, dt, joint_offsets, link_lengths, base_position):

    # Number of steps
    num_steps = int(total_time / dt) + 1
    
    # Time vector
    times = np.linspace(0, total_time, num_steps)

    # Track of joint angle values
    joint_angles = np.zeros((num_steps, 3))

    # Start joint angles
    start_joint_angles = inverse_via_DLS(start_pose, joint_offsets, link_lengths, base_position)

    # End joint angles
    end_joint_angles = inverse_via_DLS(end_pose, joint_offsets, link_lengths, base_position)

    # Raise error if IK fails
    if start_joint_angles is None or end_joint_angles is None:
        raise ValueError("IK failed for start or end pose - trajectory cannot be generated")

    # Delta joint angles
    delta_theta = end_joint_angles - start_joint_angles

    # Loop for all steps
    for i, t in enumerate(times):
        
        # Fraction of the path
        s = t / total_time

        # Piecewise linear interpolation
        joint_angles[i] = start_joint_angles + s * delta_theta

    # Return the joint angles
    return times, joint_angles


# Cubic Joint-space Trajectory
def cubic_joint_space_trajectory(start_pose, end_pose, total_time, dt, joint_offsets, link_lengths, base_position):
    # Number of steps
    num_steps = int(total_time / dt) + 1
    
    # Time vector
    times = np.linspace(0, total_time, num_steps)

    # Track of joint angle values
    joint_angles = np.zeros((num_steps, 3))

    # Start joint angles
    start_joint_angles = inverse_via_DLS(start_pose, joint_offsets, link_lengths, base_position)

    # End joint angles
    end_joint_angles = inverse_via_DLS(end_pose, joint_offsets, link_lengths, base_position)

    # Raise error if IK fails
    if start_joint_angles is None or end_joint_angles is None:
        raise ValueError("IK failed for start or end pose - trajectory cannot be generated")

    # Delta joint angles
    delta_theta = end_joint_angles - start_joint_angles

    # Set cubic polynomial coefficients 
    a0 = start_joint_angles
    a2 = (3 / total_time**2) * delta_theta
    a3 = (-2 / total_time**3) * delta_theta

    # Loop for all time steps
    for i, t in enumerate(times):

        # Cubic interpolation
        joint_angles[i] = a0 + a2*t**2 + a3*t**3

    # Return the joint angles
    return times, joint_angles

# Quintic Joint-space Trajectory
def quintic_joint_space_trajectory(start_pose, end_pose, total_time, dt, joint_offsets, link_lengths, base_position):
    # Number of steps
    num_steps = int(total_time / dt) + 1

    # Time vector
    times = np.linspace(0, total_time, num_steps)

    # Track of joint angle values
    joint_angles = np.zeros((num_steps, 3))

    # Start joint angles
    start_joint_angles = inverse_via_DLS(start_pose, joint_offsets, link_lengths, base_position)

    # End joint angles
    end_joint_angles = inverse_via_DLS(end_pose, joint_offsets, link_lengths, base_position)

    # Raise error if IK fails for start or end
    if start_joint_angles is None or end_joint_angles is None:
        raise ValueError("IK failed for start or end pose – trajectory cannot be generated")

    # Delta joint angles
    delta_theta = end_joint_angles - start_joint_angles

    # Quintic coefficients
    a0 = start_joint_angles
    a3 = 10.0 * delta_theta / (total_time**3)
    a4 = -15.0 * delta_theta / (total_time**4)
    a5 = 6.0 * delta_theta / (total_time**5)

    # Loop for all time steps
    for i, t in enumerate(times):
        
        # Quintic interpolation
        joint_angles[i] = a0 + a3 * (t**3) + a4 * (t**4) + a5 * (t**5)

    # Return the time vector and joint angles
    return times, joint_angles
