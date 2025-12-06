# src/models/state.py
import numpy as np
from copy import deepcopy

class State:
  def __init__(self, r, v, q):
    # Store position vector
    self.r = r
    # Store quaternion orientation
    self.v = v
    # Store velocity vector
    self.q = q

  def copy(self):
    # Deep copy of state
    return deepcopy(self)

  def as_vector(self):
    # State as flat vector form
    return np.concatenate((self.r, self.v, self.q))

  @staticmethod
  def from_vector(x):
    # State from flat vector
    r = x[0:3]
    v = x[3:6]
    q = x[6:10]
    return State(r, v, q)
