# kinematics/inverse.py
import numpy as np
from .forward import FK_end_effector_pose
from .jacobian import jacobian

# Dimensions
dim = 2

# Degrees of Freedom
DOF = 3

# Set position and orientation tolerance
pos_tol = 0.01
ori_tol = 0.05

# Set scaling factor
alpha = 0.5

# Lambda for DLS (squared)
DLS_lambda_sq = 0.1

# Initial estimate (zeros)
theta_initial = np.zeros(DOF)

# Cap number of iterations
max_iterations = 1000

# IK Pseudoinverse algorithm (SVD)
def inverse_via_SVD(target_pose, joint_offsets, link_lengths, base_position, theta_initial=None):
    # Track current theta
    theta_current = np.zeros(DOF) if theta_initial is None else theta_initial.copy()

    # Loop until convergence
    for i in range(max_iterations):

        # Calculate pose
        _, current_pose = FK_end_effector_pose(joint_angles=theta_current, joint_offsets=joint_offsets, link_lengths=link_lengths, base_position=base_position)

        # Set error term 
        error = target_pose - current_pose

        # Calculate Jacobian
        J = jacobian(link_lengths=link_lengths, link_angles=theta_current)

        # Calculate pseudoinverse
        J_inv = np.linalg.pinv(J)

        # Update theta
        theta_current += alpha * (J_inv @ error)

        # Calculate position and orientation error
        pos_error = np.linalg.norm(target_pose[:2] - current_pose[:2])
        ori_diff = target_pose[2] - current_pose[2]
        ori_error = abs(np.arctan2(np.sin(ori_diff), np.cos(ori_diff)))
        
        # Compare error against tolerances
        if pos_error < pos_tol and ori_error < ori_tol:
            return theta_current

    # If not converged
    raise RuntimeError(f"IK did not converge within {max_iterations} iterations.")

# Levenberg–Marquardt IK (Damped Least Squares)
def inverse_via_DLS(target_pose, joint_offsets, link_lengths, base_position, theta_initial=None):
    # Track current theta
    theta_current = np.zeros(DOF) if theta_initial is None else theta_initial.copy()

    # Loop until convergence
    for i in range(max_iterations):

        # Calculate pose
        _, current_pose = FK_end_effector_pose(joint_angles=theta_current, joint_offsets=joint_offsets, link_lengths=link_lengths, base_position=base_position)

        # Set error term 
        error = target_pose - current_pose

        # Calculate Jacobian
        J = jacobian(link_lengths=link_lengths, link_angles=theta_current)
        J_T = np.transpose(J)

        # Calculate pseudoinverse with damping term
        J_inv = J_T @ np.linalg.inv(J @ J_T + DLS_lambda_sq * np.eye(dim+1))

        # Update theta
        theta_current += alpha * (J_inv @ error)

        # Calculate position and orientation error
        pos_error = np.linalg.norm(target_pose[:2] - current_pose[:2])
        ori_diff = target_pose[2] - current_pose[2]
        ori_error = abs(np.arctan2(np.sin(ori_diff), np.cos(ori_diff)))
        
        # Compare error against tolerances
        if pos_error < pos_tol and ori_error < ori_tol:
            return theta_current

    # If not converged
    raise RuntimeError(f"IK did not converge within {max_iterations} iterations.")
