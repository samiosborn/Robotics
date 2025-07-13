# main.py 
import config
from kinematics.forward import forward_kinematics
from visualisation.plot2d import plot_2d
import numpy as np

def main():
    # Input joint angles
    input_joint_angles = np.array([np.pi/3, -np.pi/4, -np.pi/2])

    # Apply forward kinematics
    positions = forward_kinematics(
        joint_angles=input_joint_angles,
        INITIAL_JOINT_ANGLES=config.INITIAL_JOINT_ANGLES,
        LINK_LENGTHS=config.LINK_LENGTHS,
        STARTING_POSITION=config.STARTING_POSITION
    )

    # Plot in 2D
    plot_2d(positions)

if __name__ == "__main__":
    main()
