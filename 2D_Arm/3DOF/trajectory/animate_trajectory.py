# animate_trajectory.py
from kinematics.forward import forward_kinematics
from visualisation.plot2d_interactive import plot_2d
import numpy as np
import matplotlib.pyplot as plt
import time

# Animate trajectory (defined in joint angles)
def animate_angles_trajectory(trajectory_joint_angles, animation_time, joint_offsets, link_lengths, base_position):
    
    # Number of steps
    num_steps = len(trajectory_joint_angles)

    # Animation time step duration
    animation_time_step = animation_time / num_steps

    # Interactive mode on
    plt.ion()

    # Initialise figure and axes with subplots
    fig, ax = plt.subplots()

    # Loop over each joint angles in the trajectory
    for joint_angles in trajectory_joint_angles:

        # Skip invalid IK solutions
        if np.any(np.isnan(joint_angles)):
            continue

        # Forward kinematics
        joint_positions = forward_kinematics(joint_angles, joint_offsets, link_lengths, base_position)

        # Update 2D plot (interactive)
        plot_2d(joint_positions, ax)

        # Sleep
        time.sleep(animation_time_step)
    
    # Turn off interactive mode
    plt.ioff()

    # Keep the final frame
    plt.show()