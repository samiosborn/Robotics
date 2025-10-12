# simulations/single_ik_dls_sim.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config
from kinematics.inverse import inverse_via_DLS
from kinematics.forward import FK_end_effector_pose
from visualisation.plot2d import plot_2d

# Inverse kinematics example from goal pose with DLS
def single_IK_DLS():
    # Set target pose
    target_pose = np.array([0.0, 1.0, np.pi/2])

    # Estimate joint angles
    estimated_joint_angles = inverse_via_DLS(
        target_pose=target_pose,
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    # Forward kinematics from estimated joint angles
    positions, pose = FK_end_effector_pose(
        joint_angles=estimated_joint_angles,
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    print("The end effector pose is: ", np.round(pose, 3))
    print("Solved joint angles (rad):", np.round(estimated_joint_angles, 3))

    # Plot
    plot_2d(positions)

def main():
    single_IK_DLS()

if __name__ == "__main__":
    main()
