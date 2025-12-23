# src/geometry/pose.py 
import numpy as np

# Angle (radians) between translations
def angle_between_translations(t_1, t_2): 
    # Assert
    t_1 = np.asarray(t_1, dtype=float)
    t_2 = np.asarray(t_2, dtype=float)
    # Normal
    n_1 = np.linalg.norm(t_1)
    n_2 = np.linalg.norm(t_2)
    # Guard
    if n_1 < 1e-12 or n_2 < 1e-12:
        raise ValueError("Translation vectors must be non-zero")
    # Normalise
    t_1 /= n_1
    t_2 /= n_2
    # Cosine(theta)
    cos_theta = np.clip(np.dot(t_1, t_2), -1.0, 1.0)
    # Angle
    return np.arccos(cos_theta)
