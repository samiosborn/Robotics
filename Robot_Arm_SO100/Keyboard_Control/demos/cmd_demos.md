cd /Users/samiosborn/Library/CloudStorage/OneDrive-Personal/GitHub/Robotics/Robot_Arm_SO100/Keyboard_Control
conda active simple-game

# Basic keyboard teleop (default rates/gains)
python -m adapters.keyboard_control
sudo python -m adapters.keyboard_control

# Task-space, with go-to-start (+ soft-start)
# with orientation tracking (may be harder for IK)
python -m demos.move_between_waypoints_task_space --A PICK --B PLACE --go-to-start --T-start 2.0 --T 15.0 --dt 0.02
# if IK still fails, try position-only:
python -m demos.move_between_waypoints_task_space --A PICK --B PLACE --go-to-start --T-start 2.0 --T 15.0 --dt 0.02 --no-orientation

# Joint-space (linear) time-scaling, safe pre-position to start, 4 s move, 20 ms sample
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --go-to-start --T-start 2.0   --profile linear --T 4.0 --dt 0.02

# Joint-space (cubic) time-scaling, safe pre-position to start, 4 s move, 20 ms sample
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --go-to-start --T-start 2.0   --profile cubic --T 4.0 --dt 0.02

# Joint-space (quintic), safe pre-position to start, 4 s move, 20 ms sample
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --go-to-start --T-start 2.0   --profile quintic --T 4.0 --dt 0.02

