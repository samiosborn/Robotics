# src/simulations/animate_quintic_joint_space_3d_sim.py
import numpy as np
import config
from src.trajectory.joint_space_trajectory import quintic_joint_space_trajectory
from src.trajectory.animate_trajectory_3d import animate_trajectory_3d

# Animate quintic joint-space trajectory (3D)
def animate_quintic_joint_space_3d():
    start_pose = np.array([0.4, 0.7, np.pi/4])
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

    # Base rotation (entire arm spins around z)
    base_angle_start = 0.0
    base_angle_end = np.pi

    animate_trajectory_3d(
        trajectory_joint_angles,
        animation_time=config.ANIMATION_TIME,
        joint_offsets=config.JOINT_OFFSETS,
        link_lengths=config.LINK_LENGTHS,
        base_position=config.BASE_POSITION, 
        base_angle_start=base_angle_start,
        base_angle_end=base_angle_end,
    )

def main():
    animate_quintic_joint_space_3d()

if __name__ == "__main__":
    main()
