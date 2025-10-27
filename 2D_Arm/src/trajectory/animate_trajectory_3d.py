# src/trajectory/animate_trajectory_3d.py 
import numpy as np
import matplotlib.pyplot as plt
from src.kinematics.forward import forward_kinematics_3d
from src.visualisation.plot3d import plot_3d

# Animate trajectory (defined in joint angles) in 3D
def animate_trajectory_3d(trajectory_joint_angles, animation_time, joint_offsets, 
                          link_lengths, base_position, 
                          base_angles=None, base_angle_start=0.0, base_angle_end=None):
    # Number of steps
    num_steps = len(trajectory_joint_angles)

    # Duration of each animation frame
    animation_time_step = animation_time / num_steps

    # If base_angles not given, linearly interpolate between start and end
    if base_angles is None:
        if base_angle_end is None:
            base_angle_end = base_angle_start
        base_angles = np.linspace(base_angle_start, base_angle_end, num_steps)

    # Interactive mode on
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Loop over each joint angle configuration and base rotation in the trajectory
    for base_angle, joint_angles in enumerate(trajectory_joint_angles):
        # Skip invalid IK results
        if np.any(np.isnan(joint_angles)):
            continue

        # Compute forward kinematics (3D)
        joint_positions, _, _ = forward_kinematics_3d(joint_angles, joint_offsets, link_lengths, base_position, base_angle)

        # Update 3D plot (interactive)
        plot_3d(joint_positions, ax)

        # Pause for animation timing
        plt.pause(animation_time_step)

    # Turn off interactive mode
    plt.ioff()
    plt.show()
