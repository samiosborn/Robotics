# src/main.py 
import numpy as np
import config
from src.kinematics.inverse import inverse_via_SVD, inverse_via_DLS
from src.kinematics.forward import forward_kinematics, FK_end_effector_pose
from src.visualisation.plot2d import plot_2d
from src.trajectory.task_space_trajectory import linear_task_space_trajectory
from src.trajectory.animate_trajectory import animate_angles_trajectory
from src.trajectory.joint_space_trajectory import linear_joint_space_trajectory, cubic_joint_space_trajectory, quintic_joint_space_trajectory

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

# Animation of task-space linear trajectory
def animate_linear_task_space_trajectory():
    # Set start pose
    start_pose = np.array([0.4, 0.7, np.pi/4])

    # Set end pose
    end_pose = np.array([0.7, 0.4, -np.pi/4])

    times, trajectory_joint_angles = linear_task_space_trajectory(
        start_pose, 
        end_pose, 
        total_time=config.TRAJECTORY_TIME, 
        dt=config.FK_TIME_DELTA, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    animate_angles_trajectory(trajectory_joint_angles, 
        animation_time=config.ANIMATION_TIME, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

# Animation of piecewise linear task-space trajectory
def animate_linear_joint_space_trajectory():

    # Set start pose
    start_pose = np.array([0.4, 0.7, np.pi/4])

    # Set end pose
    end_pose = np.array([0.7, 0.4, -np.pi/4])

    times, trajectory_joint_angles = linear_joint_space_trajectory(
        start_pose, 
        end_pose, 
        total_time=config.TRAJECTORY_TIME, 
        dt=config.FK_TIME_DELTA, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    animate_angles_trajectory(trajectory_joint_angles, 
        animation_time=config.ANIMATION_TIME, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

# Animation of qubic task-space trajectory
def animate_cubic_joint_space_trajectory():

    # Set start pose
    start_pose = np.array([0.4, 0.7, np.pi/4])

    # Set end pose
    end_pose = np.array([0.7, 0.4, -np.pi/4])

    times, trajectory_joint_angles = cubic_joint_space_trajectory(
        start_pose, 
        end_pose, 
        total_time=config.TRAJECTORY_TIME, 
        dt=config.FK_TIME_DELTA, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    animate_angles_trajectory(trajectory_joint_angles, 
        animation_time=config.ANIMATION_TIME, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

# Animation of quintic task-space trajectory
def animate_quintic_joint_space_trajectory():

    # Set start pose
    start_pose = np.array([0.4, 0.7, np.pi/4])

    # Set end pose
    end_pose = np.array([0.7, 0.4, -np.pi/4])

    times, trajectory_joint_angles = quintic_joint_space_trajectory(
        start_pose, 
        end_pose, 
        total_time=config.TRAJECTORY_TIME, 
        dt=config.FK_TIME_DELTA, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

    animate_angles_trajectory(trajectory_joint_angles, 
        animation_time=config.ANIMATION_TIME, 
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

def main():
    # single_FK_plot() # Test FK only
    # single_IK_SVD() # Test IK via SVD
    # single_IK_DLS() # Test IK via DLS
    # animate_linear_task_space_trajectory() # Linear task-space
    # animate_linear_joint_space_trajectory() # Linear joint-space
    # animate_cubic_joint_space_trajectory() # Cubic joint-space
    animate_quintic_joint_space_trajectory() # Quintic joint-space

if __name__ == "__main__":
    main()
