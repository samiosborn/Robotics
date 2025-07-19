# Robot Kinematics

This project is to write a simple program to learn forward and inverse kinematics in 2D with a 3DOF robot arm. 

The goal is to use this understanding to extend to a 6DOF robot in 3D. 

The script will use forward kinematics (FK) to effect a movement input and determine the resulting position. 

An extension to inverse kinematics (IK) will be created to return the required inputs to reach a desired position. 

# Joint Pose

## Denavit-Hartenburg Table
First, I will define a Denavit-Hartenburg (DH) parameter matrix for the 3DOF arm. 

The DH matrix (in 3D) has 4 columns for each of the 3 joints: 
1) Rotation around z-axis
2) Translation along z-axis
3) Translation along x-axis
4) Rotation around x-axis

However, in 2D: 
- For DH column 2: No translation along z-axis (as 2D so all arms lie in xy-plane)
- For DH column 4: No rotation around x-axis (as 2D so no rotation around x-axis)

So the non-zero columns in the DH table will be: 
1) Angle of joint relative to the previous link
3) Distance between joints (arm limb lengths)

## Matrix Transformations
Each joint has a local frame defined as the positive x-axis along the length of the outward limb. 

The idea is to apply a transformation (rotation and translation) at each joint in order from origin to end-effector. 

Then, we can apply inputs at each joint, and return the xy-position and direction of the end-effector w.r.t. the origin frame. 

For plotting, it would be necessary to just know the position of the end of each joint. 

So, we are tracking two variables: the position (origin) and orientation: 
- For the position, we consider the angle of the joint, and apply a translation from the current position. 
- For the direction, we take the angle of the joint, and revolve around the current direction. 

The joint angle is measured counterclockwise from the global x-axis (positive x-direction), in radians. 

We can combine these into a single transformation matrix, T, with dimensions 3x3. 

T has two parts: 
- Top LHS is a 2x2 rotation matrix (relative to local frame)
- Right column is the translation of the origin (using homogeneous coordinates)

Homogeneous Coordinates: When there's a 1 in the bottom column. 

We will create a script to create and apply these transformation matrices based on our DH parameters and user inputs. 

## Configurations (config.py)
Link Lengths: Length in meters for each link, in order from base to end effector. 
Starting Position: Angle of each joint, in order from base to end effector. 
For example, 0 for each joint would be lying on the x-axis. 
Angle Limits: Maximum and minimum angle for each joint. 

## Denavit-Hartenburg (DH) Calculator (dh.py)
This file will compute a 2D homogeneous transform for each joint. 
So, it will compute the 3x3 matrix, composed of the 2x2 rotation matrix and origin translation. 
This is the planar equivalent of a DH transformation. 
The rotation matrix will be based on the angle applied to the joint. 
The origin will be translated the length of the outward link in the local x-direction. 
It will take as inputs the angle of the joint and length of outward link. 

# Forward Kinematics (FK)
Forward kinematics returns the resulting pose (position and orientation) given input joint angles. This is done directly by a series of transformations at each joint. 

## Forward Kinematics (forward.py)
This will apply the joint angles for each joint, and return the positions for each joint. 
Starting from the origin, return the position of each joint end, one by one. 
To adjust for the starting position, the first DH transformation should use the starting position for the origin translation. 
To adjust for the starting angle, for the rotation at each joint we must consider the combined angle to apply. 
For subsequent joints, essentially we are accumulating transformations. Then, we apply this cumulative transformation to the local origin to get the end position of the next joint. 

The transformation of the end effector is the composition equation: 

\[
T_{0 \to 3} \;=\; T(\theta_1, L_1)\; T(\theta_2, L_2)\; T(\theta_3, L_3)
\]

\[
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}
={T_{0 \to 3}}
\begin{bmatrix}
0 \\[3pt] 0 \\[3pt] 1
\end{bmatrix}
\]

## 2D Visualisation (plot2d.py)
This function takes the positions of all the ends of the joints, and plots on a 2D graph. 
Each joint is connected to the previous by a solid line. 

## Jacobian Calculator (jacobian.py)
The Jacobian is the matrix of partial derivatives.  
Thus,  
\[
J(\boldsymbol{\theta})_{ij} = \frac{\partial x_i}{\partial \theta_j}
\]  

where \(x_i\) is the \(i\)-th component of the end-effector pose \(\boldsymbol{x}\), and \(\theta_j\) is the \(j\)-th joint angle.  

Hence, for a 3-link planar manipulator:  

\[
J(\boldsymbol{\theta}) =
\begin{bmatrix}
\frac{\partial x}{\partial \theta_1} & \frac{\partial x}{\partial \theta_2} & \frac{\partial x}{\partial \theta_3} \\
\frac{\partial y}{\partial \theta_1} & \frac{\partial y}{\partial \theta_2} & \frac{\partial y}{\partial \theta_3} \\
\frac{\partial \phi}{\partial \theta_1} & \frac{\partial \phi}{\partial \theta_2} & \frac{\partial \phi}{\partial \theta_3}
\end{bmatrix}
\]

For our 2D 3DOF Robot Arm we have:  

\[
\frac{\partial x}{\partial \theta_1} = - L_1 \sin\theta_1 - L_2 \sin(\theta_1 + \theta_2) - L_3 \sin(\theta_1 + \theta_2 + \theta_3)
\]

\[
\frac{\partial x}{\partial \theta_2} = - L_2 \sin(\theta_1 + \theta_2) - L_3 \sin(\theta_1 + \theta_2 + \theta_3)
\]

\[
\frac{\partial x}{\partial \theta_3} = - L_3 \sin(\theta_1 + \theta_2 + \theta_3)
\]

Similarly,  

\[
\frac{\partial y}{\partial \theta_1} = 
L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2) + L_3 \cos(\theta_1 + \theta_2 + \theta_3)
\]

\[
\frac{\partial y}{\partial \theta_2} = 
L_2 \cos(\theta_1 + \theta_2) + L_3 \cos(\theta_1 + \theta_2 + \theta_3)
\]

\[
\frac{\partial y}{\partial \theta_3} = 
L_3 \cos(\theta_1 + \theta_2 + \theta_3)
\]

And for the orientation,  

\[
\frac{\partial \phi}{\partial \theta_1} = 1, \quad
\frac{\partial \phi}{\partial \theta_2} = 1, \quad
\frac{\partial \phi}{\partial \theta_3} = 1
\]

Which gives the full Jacobian: 
\[
\small
J(\boldsymbol{\theta}) = 
\begin{bmatrix}
-L_1 \sin\theta_1 - L_2 \sin(\theta_1 + \theta_2) - L_3 \sin(\theta_1 + \theta_2 + \theta_3) 
& -L_2 \sin(\theta_1 + \theta_2) - L_3 \sin(\theta_1 + \theta_2 + \theta_3) 
& -L_3 \sin(\theta_1 + \theta_2 + \theta_3) \\[10pt]
L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2) + L_3 \cos(\theta_1 + \theta_2 + \theta_3) 
& L_2 \cos(\theta_1 + \theta_2) + L_3 \cos(\theta_1 + \theta_2 + \theta_3) 
& L_3 \cos(\theta_1 + \theta_2 + \theta_3) \\[10pt]
1 & 1 & 1
\end{bmatrix}
\]

Our function will be able to return the Jacobian matrix given the link lengths and joint angles. 

# Inverse Kinematics (IK)

Inverse kinematics tries to find the required joint angles for a given desired end-effector pose.

If we focus on the position of the end-effector \((x, y)\):

$$
x = \sum_{k=1}^{3} L_k \cos\!\left( \sum_{i=1}^{k} \theta_i \right), 
\quad
y = \sum_{k=1}^{3} L_k \sin\!\left( \sum_{i=1}^{k} \theta_i \right)
$$

The **orientation** of the end-effector is:

$$
\phi = \theta_1 + \theta_2 + \theta_3
$$

Where:
- \(L_k\) is the length of link \(k\)  
- \(\theta_k\) is the joint angle at joint \(k\)  
- \((x, y)\) is the Cartesian position of the end-effector  
- \(\phi\) is the total orientation relative to the base frame

Let: 
- \( f \) be the forward kinematics function.  
- \( \boldsymbol{\theta} = [\theta_1, \theta_2, \theta_3] \) be the vector of joint angles  
- \( \boldsymbol{x} = [x, y, \phi] \) be the pose  

For a desired target pose \( \boldsymbol{x}^* \), we want to find \( \boldsymbol{\theta}^* \) such that:

$$
f(\boldsymbol{\theta}^*) = \boldsymbol{x}^*
$$

---

### Starting Point and Iterative Approach

Since the forward kinematics \(f\) is nonlinear, there is no simple closed-form inverse. Instead, we iteratively refine a solution for the required joint angles.

We start from an initial estimate \(\boldsymbol{\theta}_0\):  
- This could be all zeros, a previous known solution, or any educated guess close to the target.
- From this initial estimate, we update iteratively towards the desired pose.

This process is similar to the Newton–Raphson method for root finding, where: 
$$
\lim_{k \to \propto} f(\boldsymbol{\theta}_{k}) = \boldsymbol{x^*}
$$

---

### Linearisation with the Jacobian

At a current joint configuration \(\boldsymbol{\theta}_k\), a small change in joint angles \(\Delta \boldsymbol{\theta}\) produces an approximately linear change in the end-effector pose:

$$
\Delta \boldsymbol{x} \;\approx\; J(\boldsymbol{\theta}_k)\, \Delta \boldsymbol{\theta}
$$

where \(J(\boldsymbol{\theta}) = \dfrac{\partial f(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\) is the **Jacobian** matrix.

The **task-space error** between the current pose and the target is:

$$
\mathbf{e}_k = \boldsymbol{x}^* - f(\boldsymbol{\theta}_k)
$$

We want to find a joint update \(\Delta \boldsymbol{\theta}\) such that:

$$
J(\boldsymbol{\theta}_k)\, \Delta \boldsymbol{\theta} \;\approx\; \mathbf{e}_k
$$

This update is then applied to the current configuration:

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \Delta \boldsymbol{\theta}.
$$

The goal is that the new configuration \(\boldsymbol{\theta}_{k+1}\) brings the end-effector **closer** to the target, i.e.:

$$
\big\| f(\boldsymbol{\theta}_{k+1}) - \boldsymbol{x}^* \big\| \;<\; \big\| f(\boldsymbol{\theta}_{k}) - \boldsymbol{x}^* \big\|.
$$

---

### First-order Taylor expansion

Since the forward kinematics \(f\) is nonlinear, we approximate it locally using a **first-order Taylor expansion** around the current estimate \(\boldsymbol{\theta}_k\):

$$
f(\boldsymbol{\theta}_k + \Delta \boldsymbol{\theta}) \;\approx\; 
f(\boldsymbol{\theta}_k) \;+\; J(\boldsymbol{\theta}_k)\, \Delta \boldsymbol{\theta}
$$

Substituting this into the error:

$$
\mathbf{e}_{k+1} 
= \boldsymbol{x}^* - f(\boldsymbol{\theta}_{k+1})
\;\approx\;
\mathbf{e}_k - J(\boldsymbol{\theta}_k)\, \Delta \boldsymbol{\theta}
$$

So if we choose \(\Delta \boldsymbol{\theta} = J^{-1}(\boldsymbol{\theta}_k)\, \mathbf{e}_k\), we reduce the error to zero.


---

### Moore-Penrose Pseudoinverse

For most robots, the Jacobian may be non-square, redundant, or sometimes singular, so we cannot always find a true inverse. 

Instead, we solve a **least-squares problem**:

$$
\Delta \boldsymbol{\theta} = \arg\min_{\Delta \theta} \; \| J(\boldsymbol{\theta}_k)\Delta\theta - \mathbf{e}_k \|^2
$$

The solution with the *smallest joint motion* is given by the **Moore–Penrose pseudoinverse**:

$$
\Delta \boldsymbol{\theta} = J^\dagger(\boldsymbol{\theta}_k)\,\mathbf{e}_k
$$

Here, \(J^\dagger\) acts like a “generalised inverse” that maps Cartesian errors back into joint space. 

Where: 
$$
J^\dagger = J^{T} (J J^{T})^{-1}
$$

The Moore-Penrose pseudoinverse is a "right inverse" as: 

$$
J J^\dagger = J J^{T} (J J^{T})^{-1} = I
$$

---

### Intuition

- The Jacobian describes how much each joint change moves the end-effector.
- The pseudoinverse finds the minimum joint change that best matches the desired movement in task space.
- Iterating this process is like “nudging” the joint angles towards the target. 
- This is essentially Newton–Raphson for multivariable functions. 

## Inverse Kinematics via Pseudoinverse (inverse_pseudoinverse.py)
#### Iterative IK Algorithm

1. **Start with an initial estimate and define tolerance**  
   \(\boldsymbol{\theta}_0\) (e.g., all zeros or last known configuration)
   Set \( \varepsilon > 0 \)

2. **Compute the current pose**  
   \(\boldsymbol{x}_k = f(\boldsymbol{\theta}_k)\)

3. **Compute the error**  
   \(\mathbf{e}_k = \boldsymbol{x}^* - \boldsymbol{x}_k\)

4. **Compute the Jacobian**  
   \(J(\boldsymbol{\theta}_k)\)

5. **Compute the joint update**  
   \(\Delta \boldsymbol{\theta} = J^\dagger(\boldsymbol{\theta}_k)\,\mathbf{e}_k = J^{T} (J J^{T})^{-1}  \mathbf{e}_k\)

6. **Update the joint angles**  
   \(\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \Delta \boldsymbol{\theta}\) 

7. **Repeat steps 2-6 until convergence**  
   Stop when \(\|\mathbf{e}_k\| < \varepsilon\)

So the update rule is:

$$
\boxed{\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + J^\dagger(\boldsymbol{\theta}_k)\big(\boldsymbol{x}^* - f(\boldsymbol{\theta}_k)\big)}
$$

---



