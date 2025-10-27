# src/visualisation/plot3d.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to plot robot in 3D
def plot_3d(joint_positions, ax=None):
    # Ensure input is a NumPy array of shape (n_joints, 3)
    joint_positions = np.asarray(joint_positions)
    if joint_positions.shape[1] != 3:
        raise ValueError("Expected joint_positions to have shape (N, 3)")

    # Extract coordinates
    xs = joint_positions[:, 0]
    ys = joint_positions[:, 1]
    zs = joint_positions[:, 2]

    # Initialise axis if not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Clear and re-plot
    ax.cla()
    ax.plot(xs, ys, zs, '-o', color='b', linewidth=3, markersize=6)

    # Compute dynamic axes limits based on arm reach
    reach = np.max(np.linalg.norm(joint_positions, axis=1)) * 1.2
    ax.set_xlim([-reach, reach])
    ax.set_ylim([-reach, reach])
    ax.set_zlim([0, reach])

    # Labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3-DOF Planar Arm in 3D Space")

    # Maintain consistent aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Keep the view stable across frames
    ax.view_init(elev=25, azim=45)

    # Update the plot interactively
    plt.draw()
    plt.pause(0.001)

    return ax
