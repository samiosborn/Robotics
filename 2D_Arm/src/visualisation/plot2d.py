# src/visualisation/plot2d.py
import matplotlib.pyplot as plt

# Function to plot robot in 2D
def plot_2d(joint_positions):
    # Initialise figure and axes with subplots
    fig, ax = plt.subplots()

    # Set title
    ax.set_title("3DOF 2D FK Simulation")

    # Set aspect ratio
    ax.set_aspect('equal')

    # Show x,y axes at origin
    ax.axhline(0, color = 'grey', linewidth = 2)
    ax.axvline(0, color = 'grey', linewidth = 2)

    # Set x and y limits for axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Plot using line graph
    ax.plot(joint_positions[:,0], joint_positions[:,1], marker='o', color='b', linewidth = 4)

    # Show plot
    plt.show()
