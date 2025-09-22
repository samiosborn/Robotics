# ---- quick kinematics tests ----
python -m tests.kin_fk_01_shoulder_insert_smoke
python -m tests.kin_fk_02_shoulder_circle_pan_sweep
python -m tests.kin_fk_03_wrist_reach_straight
python -m tests.kin_fk_04_numeric_dq1_jacobian
python -m tests.kin_fk_05_toggle_offset_effect

# ---- joint-space demo: DRY RUNS (PICK -> PLACE) ----
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --dry-run
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --profile quintic --T 3.5 --dt 0.02 --dry-run
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --profile cubic   --T 4.0 --dry-run

# warn-if-far check (no pre-position). still a dry run.
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --T 3.0 --dt 0.02 --dry-run

# include pre-position to start A (recommended on hardware), still dry-run preview
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --go-to-start --T-start 2.0 --T 4.0 --dt 0.02 --dry-run

# try soft-start settings (conditional pre-position), dry-run preview
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --profile quintic --T 4.0 --dt 0.02 --soft-start --soft-start-T 1.0 --soft-start-tol-deg 0.3 --dry-run

# disable soft-start (for comparison): just omit soft-start flags
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --profile quintic --T 3.0 --dt 0.02 --dry-run

# reverse direction (PLACE -> PICK), dry-run
python -m demos.move_between_waypoints_joint_space --A PLACE --B PICK --profile quintic --T 4.0 --dt 0.02 --dry-run

# ---- joint-space demo: LIVE RUNS (remove --dry-run; run ONE at a time) ----
python -m demos.move_between_waypoints_joint_space --A PICK  --B PLACE --go-to-start --T-start 2.0 --profile quintic --T 4.0 --dt 0.02
python -m demos.move_between_waypoints_joint_space --A PLACE --B PICK  --go-to-start --T-start 2.0 --profile quintic --T 4.0 --dt 0.02
python -m demos.move_between_waypoints_joint_space --A PICK  --B PLACE --profile quintic --T 4.0 --dt 0.02 --soft-start --soft-start-T 1.0 --soft-start-tol-deg 0.3

# ---- task-space demo: DRY RUNS ----
python -m demos.move_between_waypoints_task_space --A PICK --B PLACE --T 4.0 --dt 0.02 --dry-run
python -m demos.move_between_waypoints_task_space --A PICK --B PLACE --go-to-start --T-start 2.0 --T 5.0 --dt 0.02 --dry-run

# ---- task-space demo: LIVE RUNS (run ONE at a time) ----
python -m demos.move_between_waypoints_task_space --A PICK  --B PLACE --go-to-start --T-start 2.0 --T 5.0 --dt 0.02
python -m demos.move_between_waypoints_task_space --A PLACE --B PICK  --go-to-start --T-start 2.0 --T 5.0 --dt 0.02

# ---- edge / stress checks (dry-run) ----
# coarser dt
python -m demos.move_between_waypoints_joint_space --A PICK --B PLACE --profile linear --T 3.0 --dt 0.05 --dry-run
# longer move
python -m demos.move_between_waypoints_task_space  --A PICK --B PLACE --T 8.0 --dt 0.02 --dry-run
