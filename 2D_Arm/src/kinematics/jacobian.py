# src/kinematics/jacobian.py
import numpy as np

DOF = 3

# Calculate the Jacobian matrix
def jacobian(link_lengths: np.ndarray, joint_angles: np.ndarray, joint_offsets: np.ndarray):
    # Initialise the matrix with zeros
    J = np.zeros((DOF,DOF), dtype=float)

    # Constants
    l1, l2, l3 = link_lengths
    t1, t2, t3 = joint_angles
    s1 = np.sin(t1)
    s12 = np.sin(t1+t2)
    s123 = np.sin(t1+t2+t3)
    c1 = np.cos(t1)
    c12 = np.cos(t1+t2)
    c123 = np.cos(t1+t2+t3)

    # Partial derivatives of x
    J[0,0]=-l1*s1-l2*s12-l3*s123
    J[0,1]=-l2*s12-l3*s123
    J[0,2]=-l3*s123

    # Partial derivatives of y
    J[1,0]=l1*c1+l2*c12+l3*c123
    J[1,1]=l2*c12+l3*c123
    J[1,2]=l3*c123

    # Partial derivatives of phi (orientation)
    J[2, :] = 1.0

    return J