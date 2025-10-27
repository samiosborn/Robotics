# src/simulations/animate_linear_task_space_sim.py
import numpy as np
import config
from src.trajectory.task_space_trajectory import linear_task_space_trajectory
from src.trajectory.animate_trajectory import animate_angles_trajectory

# Animate linear task-space trajectory
def animate_linear_task_space():
    start_pose = np.array([0.4, 0.7, np.pi/4])
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

    animate_angles_trajectory(
        trajectory_joint_angles,
        animation_time=config.ANIMATION_TIME,
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION
    )

def main():
    animate_linear_task_space()

if __name__ == "__main__":
    main()
