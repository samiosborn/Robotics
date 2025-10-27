# src/dynamics/spatial_transforms.py
import numpy as np
import config
from src.dynamics.spatial_math import skew3, vec3_from_skew

# Rotation matrix (3x3) around z-axis by angle theta
def rotate3z(theta: float): 
    # (x,y) components
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])

# Rotation matrix (3x3) from Euler angles (yaw, pitch, roll)
def rotation_matrix_from_euler(yaw=0.0, pitch=0.0, roll=0.0):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])
    return Rz @ Ry @ Rx

# Rotation matrix (6x6) around z-axis by angle theta for spatial motion
def rotate6z(theta: float):
    # Rotation matrix (3x3)
    R3 = rotate3z(theta)
    X = np.zeros((6, 6), dtype=float)
    X[:3, :3] = R3
    X[3:, 3:] = R3
    return X

# Translation (6x6) from 3-vector for spatial motion
def translate6(r3: np.ndarray):
    # Skew matrix (3x3)
    S = skew3(r3)
    X = np.eye(6, dtype=float)
    X[3:6, 0:3] = S
    return X
    
# Screw spatial screw transform (6x6) around z-axis 
def screw6z():
    S = np.zeros((6,1), dtype=float)
    S[2, 0] = 1.0
    return S

# Parent array for a serial open-chain of length n
def parents_serial(n: int):
    # Base is position -1
    return np.array([-1] + list(range(n-1)), dtype=int)

# Build spatial motion transform X for parent to child for a planar serial chain from joint angles
def build_chain_transforms(q: np.ndarray, joint_offsets: np.ndarray, link_lengths: np.ndarray):
    # Chain length
    n = len(q)
    # Create parent array of indices
    parent = parents_serial(n)
    # Initialise spatial transform up
    X_up = []
    # Initialise list of joint screw transforms
    S_list = []
    # Loop over each joint
    for i in range(n):
        # Angle
        theta_i = float(q[i] + joint_offsets[i])
        # Rotate joint in z-axis
        R6z = rotate6z(theta_i)
        # Link geometry
        r_i = np.array([link_lengths[i], 0.0, 0.0], dtype=float)
        # Translation along link length in x-axis
        Xt = translate6(r_i)
        # Spatial transform: rotate, then translate
        Xi = Xt @ R6z
        # Screw transform (in child frame)
        Si = screw6z()
        # Append
        S_list.append(Si)
        X_up.append(Xi)
    return X_up, S_list, parent    

# Extract rotation matrix and translation vector from spatial transform 
def extract_rotation_translation_from_spatial_transform(X: np.ndarray):
    # Rotation matrix
    R = X[:3, :3]
    # Screw transform
    S = X[3:, :3]
    # Translation vector
    r = vec3_from_skew(S)
    return R, r

# Build spatial transform from rotation and translation
def build_spatial_transform_from_rotation_translation(R: np.ndarray, r: np.ndarray):
    # Allocate empty spatial transform
    X = np.zeros((6, 6), dtype=float)
    # Rotation
    X[:3, :3] = R
    X[3:, 3:] = R
    # Convert translation to skew-symmetric
    S = skew3(r)
    # Translation
    X[3:, :3] = S
    return X

# Build homogenous world pose (4x4) from rotation matrix and translation
def world_pose_from_rotation_translation(R: np.ndarray, r: np.ndarray):
    # Allocate homogeneous pose in 3D
    T = np.eye(4)
    # Rotation
    T[:3, :3] = R
    # Translation
    T[:3, 3] = r
    return T

# World SE(3) poses from spatial motion poses
def world_poses_from_X(X_up: list[np.ndarray], parent: np.ndarray, base_position: np.ndarray):
    n = len(X_up)
    T_base = np.eye(4, dtype=float)
    T_base[:2, 3] = base_position
    # World poses
    T_world = [np.eye(4, dtype=float) for _ in range(n)]
    for i in range(n):
        X = X_up[i]
        # Extract (R, r) from planar motion transform: X = [[R,0],[S,R]], S=skew(r)
        R = X[:3, :3]
        S = X[3:6, 0:3]
        r = vec3_from_skew(S)
        # Build homogeneous pose
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3,  3] = r
        # 4x4 SE(3) pose of link i in world
        T_world[i] = (T_base @ T) if parent[i] == -1 else (T_world[parent[i]] @ T)
    return T_world

# Return world pose transformations for each link from joint angles
def world_poses_from_q(q: np.ndarray, joint_offsets: np.ndarray, 
                       link_lengths:  np.ndarray, base_position: np.ndarray):
    # Assertions
    if joint_offsets is None: joint_offsets = config.JOINT_OFFSETS
    if link_lengths is None: link_lengths = config.LINK_LENGTHS
    if base_position is None: base_position = config.BASE_POSITION
    # Build spatial transform
    X_up, _, parent = build_chain_transforms(q, joint_offsets, link_lengths)
    # Return world poses from spatial transform
    return world_poses_from_X(X_up, parent, base_position)

# World positions and COMs
def world_positions_coms(q: np.ndarray, joint_offsets: np.ndarray, 
                         link_lengths: np.ndarray, base_position: np.ndarray, links):
    # Assertions
    if joint_offsets is None: joint_offsets = config.JOINT_OFFSETS
    if link_lengths is None: link_lengths = config.LINK_LENGTHS
    if base_position is None: base_position = config.BASE_POSITION
    # World poses
    T_list = world_poses_from_q(q, joint_offsets, link_lengths, base_position)
    # Number of links/joints
    n = len(T_list)
    assert len(links) == n, "links list length must match number of joints"
    # World positions of joint axes (link origin)
    r_joint = np.zeros((n, 3), dtype=float)
    # World position of link COMs
    r_com = np.zeros((n, 3), dtype=float)
    # Loop over each joint
    for i, T in enumerate(T_list): 
        # Rotation
        R = T[:3, :3]
        # Position
        p = T[:3, 3]
        r_joint[i] = p
        # COM in local frame
        com_local = np.array([links[i].com_xy[0], links[i].com_xy[1], 0.0], dtype=float)
        # COM in world
        r_com[i] = p + R @ com_local
    return r_joint, r_com
