# simulations/single_ik_svd_sim.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config
from kinematics.inverse import inverse_via_SVD
from kinematics.forward import FK_end_effector_pose
from visualisation.plot2d import plot_2d

# Inverse kinematics example from goal pose with SVD
def single_IK_SVD():
    # Set target pose
    target_pose = np.array([0.7, 0.4, -np.pi/4])

    # Estimate joint angles
    estimated_joint_angles = inverse_via_SVD(
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
    single_IK_SVD()

if __name__ == "__main__":
    main()
