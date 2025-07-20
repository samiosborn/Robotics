# main.py 
import config
from kinematics.inverse import inverse_via_SVD, inverse_via_DLS
from kinematics.forward import forward_kinematics, FK_end_effector_pose
from visualisation.plot2d import plot_2d
import numpy as np

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

# Inverse example from goal pose with SVD
def single_IK_SVD():
    # Set target pose
    target_pose = np.array([0.7, 0.4, -np.pi/4]) 

    # Estimated link angles
    estimated_link_angles = inverse_via_SVD(
        target_pose=target_pose, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    # Run FK on the estimated link angles
    positions, pose = FK_end_effector_pose(
        joint_angles=estimated_link_angles,
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    print("The end effector pose is: ", np.round(pose,3))
    print("Solved joint angles (rad):", np.round(estimated_link_angles, 3))

    # Plot in 2D
    plot_2d(positions)

# Inverse example from goal pose with DLS
def single_IK_DLS():
    # Set target pose
    target_pose = np.array([0.0, 1.0, np.pi/2]) 

    # Estimated link angles
    estimated_link_angles = inverse_via_DLS(
        target_pose=target_pose, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    # Run FK on the estimated link angles
    positions, pose = FK_end_effector_pose(
        joint_angles=estimated_link_angles,
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    print("The end effector pose is: ", np.round(pose,3))
    print("Solved joint angles (rad):", np.round(estimated_link_angles, 3))

    # Plot in 2D
    plot_2d(positions)

def main():
    # single_FK_plot() # Test FK only
    # single_IK_SVD() # Test IK via SVD
    single_IK_DLS() # Test IK via DLS

if __name__ == "__main__":
    main()
