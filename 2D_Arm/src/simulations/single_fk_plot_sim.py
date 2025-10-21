# simulations/single_fk_plot_sim.py
import numpy as np
import config
from kinematics.forward import forward_kinematics
from visualisation.plot2d import plot_2d

# Plot a single FK action on a graph
def single_FK_plot():
    # Input joint angles
    input_joint_angles = np.array([np.pi/3, -np.pi/4, -np.pi/2])

    # Apply forward kinematics
    positions = forward_kinematics(
        joint_angles=input_joint_angles,
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    # Plot in 2D
    plot_2d(positions)

def main():
    single_FK_plot()

if __name__ == "__main__":
    main()
