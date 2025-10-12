# Robot Kinematics

This project is to write a simple program to learn forward and inverse kinematics in 2D with a 3DOF robot arm. 

The goal is to use this understanding to extend to a 6DOF robot in 3D. 

The script will use forward kinematics (FK) to effect a movement input and determine the resulting position. 

An extension to inversae kinematics (IK) will be created to return the required inputs to reach a desired position. 

## Joint Pose

### Denavit-Hartenburg Table
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

### Matrix Transformations
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

### Configurations (config.py)
Link Lengths: Length in meters for each link, in order from base to end effector. 
Starting Position: Angle of each joint, in order from base to end effector. 
For example, 0 for each joint would be lying on the x-axis. 
Angle Limits: Maximum and minimum angle for each joint. 

### Denavit-Hartenburg (DH) Calculator (dh.py)
This file will compute a 2D homogeneous transform for each joint. 
So, it will compute the 3x3 matrix, composed of the 2x2 rotation matrix and origin translation. 
This is the planar equivalent of a DH transformation. 
The rotation matrix will be based on the angle applied to the joint. 
The origin will be translated the length of the outward link in the local x-direction. 
It will take as inputs the angle of the joint and length of outward link. 

## Forward Kinematics (FK)
Forward kinematics returns the resulting pose (position and orientation) given input joint angles. This is done directly by a series of transformations at each joint. 

### Forward Kinematics (forward.py)
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

### 2D Visualisation (plot2d.py)
This function takes the positions of all the ends of the joints, and plots on a 2D graph. 
Each joint is connected to the previous by a solid line. 

### Jacobian Calculator (jacobian.py)
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
\(
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
\)

Our function will be able to return the Jacobian matrix given the link lengths and joint angles. 

## Inverse Kinematics (IK)

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

#### Starting Point and Iterative Approach

Since the forward kinematics \(f\) is nonlinear, there is no simple closed-form inverse. Instead, we iteratively refine a solution for the required joint angles.

We start from an initial estimate \(\boldsymbol{\theta}_0\):  
- This could be all zeros, a previous known solution, or any educated guess close to the target.
- From this initial estimate, we update iteratively towards the desired pose.

This process is similar to the Newton–Raphson method for root finding, where: 
$$
\lim_{k \to \propto} f(\boldsymbol{\theta}_{k}) = \boldsymbol{x^*}
$$

---

#### Linearisation with the Jacobian

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

#### First-order Taylor expansion

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

#### Moore-Penrose Pseudoinverse

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

#### Iterative IK Algorithm

1. Start with an initial estimate and define tolerance
   \(\boldsymbol{\theta}_0\) (e.g., all zeros or last known configuration)
   Set \( \varepsilon > 0 \)

2. Compute the current pose
   \(\boldsymbol{x}_k = f(\boldsymbol{\theta}_k)\)

3. Compute the error 
   \(\mathbf{e}_k = \boldsymbol{x}^* - \boldsymbol{x}_k\)

4. Compute the Jacobian
   \(J(\boldsymbol{\theta}_k)\)

5. Compute the joint update
   \(\Delta \boldsymbol{\theta} = J^\dagger(\boldsymbol{\theta}_k)\,\mathbf{e}_k = J^{T} (J J^{T})^{-1}  \mathbf{e}_k\)

6. Update the joint angles 
   \(\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \Delta \boldsymbol{\theta}\) 

7. Repeat steps 2-6 until convergence
   Stop when \(\|\mathbf{e}_k\| < \varepsilon\)

So the update rule is:

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + J^\dagger(\boldsymbol{\theta}_k)\big(\boldsymbol{x}^* - f(\boldsymbol{\theta}_k)\big)
$$

#### Intuition

- The Jacobian describes how much each joint change moves the end-effector.
- The pseudoinverse finds the minimum joint change that best matches the desired movement in task space.
- Iterating this process is like “nudging” the joint angles towards the target. 
- This is essentially a Jacobian-based Gauss–Newton method. 

---

### Avoiding Singularities

Singularities occur in the Jacobian \( J \) matrix when it can't be inverted. This is where one dimension collapses. 

For an example of this, consider a 2DOF 2D Robot arm with joint angles \( \theta_1, \theta_2 \). Suppose the arm is straight line when \( \theta_2 = 0 \). 
Then, any small change in \( \theta_2 \) will move the end effector in an arc that's tangent is perpendicular to the line. Similarly with a small change to \( \theta_1 \). 
Thus, changes to both joint angles are parallel in effect and thus the rank of J drops by 1 and it's no longer invertible. 

#### Singular Value Decomposition (SVD)

To deal with singularities when inverting \( (J J^{T})^{-1} \), we use Singular Value Decomposition (SVD). 

For any real \(m \times n\) matrix \(J\), there always exist:  

- \(U \in \mathbb{R}^{m \times m}\): an orthogonal matrix  
- \(V \in \mathbb{R}^{n \times n}\): an orthogonal matrix  
- \(\Sigma \in \mathbb{R}^{m \times n}\): a diagonal-like matrix with non-negative entries  

such that  

\[
\boxed{J = U \, \Sigma \, V^\top}
\]

\(\Sigma = \mathrm{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)\) where:

- \(\sigma_i \ge 0\) are the **singular values** of \(J\)  
   - They are the square roots of the eigenvalues of \(J^\top J\) (or \(J J^\top\))  

#### Singular Values

For \(J\), two important features of \(J^\top J\): 

1. Symmetric:
\[
(J^\top J)^\top = J^\top (J^\top)^\top = J^\top J.
\]

2. Positive semi-definite:* For any vector \(x \in \mathbb{R}^n\),  
\[
x^\top (J^\top J)\, x = (Jx)^\top (Jx) = \|Jx\|^2 \ge 0.
\]

Recall the definition for an eigenvector \(v\) and eigenvalue \(\lambda\) of \(J^\top J\):  

\[
J^\top J v = \lambda v.
\]

Now pre-multiply by \(v^\top\):  

\[
v^\top (J^\top J) v = \lambda\, v^\top v.
\]

But \(v^\top (J^\top J) v = \|Jv\|^2 \ge 0\).  
Also \(v^\top v = \|v\|^2 > 0\) for any nonzero eigenvector.  

Therefore:  

\[
\lambda \|v\|^2 = \|Jv\|^2 \ge 0 \quad \Rightarrow \quad \lambda \ge 0.
\]

Hence all eigenvalues \(\lambda_i\) of \(J^\top J\) are non-negative.

The singular values \(\sigma_i\) of \(J\) are then defined as: 

\[
\sigma_i = \sqrt{\lambda_i} \ge 0.
\]

---

#### Geometric Understanding

Multiplication by \(J\) maps a unit sphere in \(\mathbb{R}^n\) to an ellipsoid in \(\mathbb{R}^m\).  
The lengths of the ellipsoid’s principal axes are exactly the singular values \(\sigma_i\),  which must be non-negative (no “negative length”).

---

#### Identification of Singularities

The Singular Value Decomposition  

\[
J = U \Sigma V^\top
\]

satisfies  

\[
J^\top J = V \Sigma^2 V^\top, 
\quad 
J J^\top = U \Sigma^2 U^\top.
\]

Thus the non-zero eigenvalues of \(J^\top J\) (and \(J J^\top\)) are the same,  
and their square roots are the singular values \(\sigma_i\).

Rank and singularity: 
- If all \(\sigma_i > 0\), then \(J\) is full rank → invertible (if square).  
- If some \(\sigma_i = 0\), rank drops → singularity.  

Thus, SVD provides a numerical way to detect singularities by inspecting the singular values \(\sigma_i\).  


Given the SVD \(J = U \Sigma V^\top\), the Moore–Penrose pseudoinverse is  

\[
J^\dagger = V \, \Sigma^\dagger U^\top \\
\Sigma^\dagger = \mathrm{diag}\!\left( \frac{1}{\sigma_i} \;\text{, if } \sigma_i \neq 0,\; 0 \text{ otherwise} \right)
\]

- For large \(\sigma_i\) --> normal inversion  
- For \(\sigma_i \to 0\) --> division blows up → numerical instability near singularity  

Hence SVD pseudoinverse does not avoid singularities — it still explodes when \(\sigma_i \approx 0\)

---

#### Damped Least Squares (DLS)

Damped Least Squares introduces a damping term \( \lambda  > 0 \) which avoids singularities. 

In the Moore-Penrose pseudoinverse method, we have:
$$
J^\dagger = J^{T} (J J^{T})^{-1}
$$

In DLS instead we have: 
$$
J^\dagger_\lambda = J^{T} (J J^{T} + \lambda^2 I)^{-1}
$$

Recall from the SVD of the Jacobian \(J\):
\[
J = U \Sigma V^\top, 
\quad \Sigma = \mathrm{diag}(\sigma_1, \sigma_2, \dots)
\]

the damped pseudoinverse is defined by modifying the singular value inversion:

Using the SVD \(J = U \Sigma V^\top\), since \(V\) is orthogonal, we have \( V^\top V = I \)

So it cancels out in the product:

\[
J J^\top 
= U \Sigma V^\top \; (U \Sigma V^\top)^\top
= U \Sigma (V^\top V) \Sigma^\top U^\top
= U \,\Sigma \Sigma^\top\, U^\top.
\]

\[
J J^\top = U \Sigma^2 U^\top
\]

Since \( U \) is orthogonal:
\[
J J^\top + \lambda^2 I = U \left( \Sigma^2 + \lambda^2 I \right) U^\top
\]

Its inverse is:
\[
(J J^\top + \lambda^2 I)^{-1} 
= U \, \mathrm{diag}\!\left( \frac{1}{\sigma_i^2 + \lambda^2} \right) U^\top.
\]

Now multiply:
\[
\begin{aligned}
J^\top (J J^\top + \lambda^2 I)^{-1} 
&= (V \Sigma U^\top)^\top \; U \, \mathrm{diag}\!\left( \tfrac{1}{\sigma_i^2 + \lambda^2} \right) U^\top \\
&= V \Sigma^\top U^\top U \, \mathrm{diag}\!\left( \tfrac{1}{\sigma_i^2 + \lambda^2} \right) U^\top \\
&= V \, \mathrm{diag}\!\left( \frac{\sigma_i}{\sigma_i^2 + \lambda^2} \right) U^\top
\end{aligned}
\]

\[
J^\dagger_\lambda 
= V \, \mathrm{diag}\!\!\left( \frac{\sigma_i}{\sigma_i^2 + \lambda^2} \right) U^\top
\]

Which modifies the singular value inversion:  

\[
\frac{1}{\sigma_i} \;\;\longrightarrow\;\; \frac{\sigma_i}{\sigma_i^2 + \lambda^2}
\]



- If \(\sigma_i \gg \lambda\), behaves like \(1/\sigma_i\) (normal pseudoinverse)  
- If \(\sigma_i \to 0\), becomes \(\approx \sigma_i / \lambda^2\) → finite, no blow-up  

This regularisation avoids instability near singularities.

Reminder that this singularity is transient and can normally be overcome. 
The issue is when near-infinite joint torques are being applied to joint servos in traditional SVD. 

### Inverse Kinematics (inverse.py)

This function executes the IK algorithm until convergence. 

# Trajectory Planning

Trajectory planning is how we plan to move the end effector pose from one position to another. 
This problem is divided into two methods: 
- Task-Space based: This is the Cartesian space where the end effector pose may exist. 
- Joint-Space based: This concerns changing joint angles within their respective limits. 

## Task-Space Trajectories: 

Suppose we have a start pose \( \boldsymbol{x_0} = (x_0, y_0, \phi_0) \) and end pose \( \boldsymbol{x_T} = (x_T, y_T, \phi_T) \)

We can consider the linear interpolation in task-space \( l(t) \) with \( t: 0 \to T \) where: 
\(
l(t) = 
\begin{bmatrix}
x_0 + \frac{t}{T} (x_T - x_0) \\
y_0 + \frac{t}{T} (y_T - y_0)\\
\phi_0 + \frac{t}{T} ( \phi_T - \phi_0)
\end{bmatrix}
\)

This is simply a straight line in task space. 

Discretise the motion with a time interval step \( dt \), then for every step: 
1. For step \( k \), compute the pose \( l(t_k) \) at time \( t * dt\)
2. Use inverse kinematics to return the joint angles \( \boldsymbol{\theta_k} \) required for that pose
3. Send the joints to those required joint angles

#### Positives
- End effector moves in a continuous linear path in Cartesian space. 
- Likely moves in the shortest possible path (a straight line), using less time and energy. 
- Intuitive for tasks which are straight: welding joints or drawing lines. 

#### Negatives
- Computationally heavy: requires calculating the IK at each time step. 
- IK may fail due to singularities or joint limits mid-path. 
- IK may have multiple solutions causing joint angle jumps. 

### Linear Task-Space Trajectory (task_space_trajectory.py)

This program executes the linear task-space trajectory planner. 

At each time step, the program returns the required joint angles for the forward kinematics. 

The function takes in the start and end pose, as well as a total time $T$ and time step $dt$ 

### Animate Trajectory (animate_trajectory.py)

This program animates the trajectory given by a series of joint angles, by running forward kinematics at each time step, and calling the plot function on an interactive basis. 

## Joint-Space Trajectories (joint_space_trajectory.py)

Joint-Space trajectories interpolate in joint-space, this is the space in which joint values exist. 

Suppose we have a start pose \( \boldsymbol{x_0} = (x_0, y_0, \phi_0) \) which corresponds to start joint angles \( \boldsymbol{\theta_0} = (\theta_{1,0}, \theta_{2,0}, \theta_{3,0} ) \) 
With end pose \( \boldsymbol{x_T} = (x_T, y_T, \phi_T) \) which corresponds to start joint angles \( \boldsymbol{\theta_T} = (\theta_{1,T}, \theta_{2,T}, \theta_{3,T} ) \)

We can consider the linear interpolation in joint-space \( \theta(t) \) with \( t: 0 \to T \) where: 

\(
\theta(t) = 
\begin{bmatrix}
\theta_{1,0} + \frac{t}{T} (\theta_{1,T} - \theta_{1,0}) \\
\theta_{2,0} + \frac{t}{T} (\theta_{2,T} - \theta_{2,0}) \\
\theta_{3,0} + \frac{t}{T} (\theta_{3,T} - \theta_{3,0})
\end{bmatrix}
\)

In this piecewise linear case, we have the boundary conditions satisfied on the joint angles: \( \theta(0) = \theta_0 \) and \( \theta(T) = \theta_T \)

Now this trajectory will be smooth in joint-space, however will likely be curved in Cartesian task-space. 

#### Positives
- Does not require computing the inverse kinematics every step, light to run. 
- Ensures joint angles stay within possible bounds. 
- No jumping around like in task-space trajectories (no ambiguity as to the next joint angle input). 

#### Negatives
- End effector likely will not move in a straight line in task-space. 
- Trajectory may not be predicatable, and may not have the lowest time and energy cost. 
- Piecewise linear interpolation has a jump in velocity, which causes extreme acceleration, which is bad for joint motors. 

### Cubic Joint-Space Trajectory

The problem with the piecewise linear joint-space trajectory is the discontinuity in velocity - as the velocity jumps at \( t = 0 \), and also drops to zero at \( t = T \). 

Instead, let's impose boundary conditions for the joint angular velocity: \( \dot{\theta} (0) = 0\) and \( \dot{\theta} (T) = 0 \)

Now 4 boundary conditions creates a polynomial solution with 4 unknowns - so a qubic. 

Let's define an arbitrary qubic: 

\( \theta(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 \)

Given \( \theta(0) = \theta_0 \), we have \( a_0 = \theta_0 \) and thus \( \theta(t) = \theta_0 + a_1 t + a_2 t^2 + a_3 t^3 \)

Differentiating: 
\( \dot{\theta}(t) = a_1 + 2 a_2 t + 3 a_3 t^2\)

Given \( \dot{\theta} (0) = 0 \), then \( a_1 = 0 \) so \( \theta(t) = \theta_0 + a_2 t^2 + a_3 t^3  \)

Restating: \( \dot{\theta}(t) = 2 a_2 t + 3 a_3 t^2\)

Using \( \dot{\theta} (T) = 0 \), \( 0 = 2 a_2 T + 3 a_3 T^2 = T (2 a_2 + 3a_3 T) \)

Since \( T \neq 0\), \( a_3 = \frac{- 2 a_2}{3 T} \)

Restating: \( \theta(t) = \theta_0 + a_2 t^2 + \frac{- 2 a_2}{3 T}  t^3  \)

Given \( \theta(T) = \theta_T \) we have \( \theta_T = \theta_0 + a_2 T^2 + \frac{- 2 a_2}{3 T}  T^3 = \theta_0 + a_2 T^2 + \frac{- 2 a_2}{3}  T^2  \)

Simplified:  \( \theta_T - \theta_0 = a_2  \frac{T^2}{3}  \) thus \( a_2 = \frac{3 (\theta_T - \theta_0 )}{T^2} \)

In conclusion: \( \theta(t) = \theta_0 + \frac{3 (\theta_T - \theta_0 )}{T^2} t^2 - \frac{2 (\theta_T - \theta_0 )}{T^3}  t^3  \)

### Quintic Joint-Space Trajectory

The problem with the cubic joint-space trajectory is that while it ensures zero velocity at the start and end, the acceleration still jumps at \( t = 0 \) and \( t = T \).  

Instead, let’s also impose boundary conditions for the joint angular acceleration:  
\[
\ddot{\theta}(0) = 0, \quad \ddot{\theta}(T) = 0
\]

Now 6 boundary conditions create a polynomial solution with 6 unknowns – so a quintic.  

Let’s define an arbitrary quintic:  

\[
\theta(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5
\]

Differentiating:  
\[
\dot{\theta}(t) = a_1 + 2 a_2 t + 3 a_3 t^2 + 4 a_4 t^3 + 5 a_5 t^4
\]
\[
\ddot{\theta}(t) = 2 a_2 + 6 a_3 t + 12 a_4 t^2 + 20 a_5 t^3
\]

Given \( \theta(0) = \theta_0 \), we have \( a_0 = \theta_0 \) 
Given \( \dot{\theta}(0) = 0 \), then \( a_1 = 0 \)
Given \( \ddot{\theta}(0) = 0 \), then \( a_2 = 0 \)  

So now:  \( \theta(t) = \theta_0 + a_3 t^3 + a_4 t^4 + a_5 t^5  \)

Using the final boundary conditions at \( t = T \):  

\[
\theta_T = \theta_0 + a_3 T^3 + a_4 T^4 + a_5 T^5
\]
\[
0 = 3 a_3 T^2 + 4 a_4 T^3 + 5 a_5 T^4
\]
\[
0 = 6 a_3 T + 12 a_4 T^2 + 20 a_5 T^3
\]

Let \( \Delta\theta = \theta_T - \theta_0 \)
Solving gives:  

\[
a_3 = \frac{10 \Delta\theta}{T^3}, \quad a_4 = -\frac{15 \Delta\theta}{T^4}, \quad a_5 = \frac{6 \Delta\theta}{T^5}
\]

In conclusion:  \( \theta(t) = \theta_0 + \frac{10 (\theta_T - \theta_0)}{T^3} t^3 - \frac{15 (\theta_T - \theta_0)}{T^4} t^4 + \frac{6 (\theta_T - \theta_0)}{T^5} t^5 \)

Or in normalised form:  \( \theta(t) = \theta_0 + \Delta\theta \Big[ 10 \left(\frac{t}{T}\right)^3 - 15 \left(\frac{t}{T}\right)^4 + 6 \left(\frac{t}{T}\right)^5 \Big] \)

# Noise

Noise represents uncertainty in our model and measurements. We generally distinguish between two main types:

## Process Noise

Process noise represents uncertainty in the system dynamics. It's likely our model is not a perfect representation so the noise reflects any unmodelled friction or actuator errors for example. 

If our discrete-time process model is:

\[
\mathbf{x}_{t+1} = A_t \, \mathbf{x}_t + B_t \, \mathbf{u}_t + \mathbf{w}_t
\]

Then:

- \( \mathbf{x}_t \in \mathbb{R}^n \) is the state vector at time \(t\)  
- \( \mathbf{u}_t \) is a control input  
- \( \mathbf{w}_t \) is the process noise, typically modelled as zero-mean Gaussian:

\[
\mathbf{w}_t \sim \mathcal{N}(0, Q_t)
\]

Where:

- \( Q_t \in \mathbb{R}^{n \times n} \) is the process noise covariance matrix

---

## Observation Noise

Observation noise represents uncertainty in the measurements we take from sensors.

If our measurement model is:

\[
\mathbf{y}_t = H_t \, \mathbf{x}_t + \mathbf{v}_t
\]

then:

- \( \mathbf{y}_t \in \mathbb{R}^m \) is the measurement vector
- \( \mathbf{v}_t \) is the observation noise, typically modelled as zero-mean Gaussian:

\[
\mathbf{v}_t \sim \mathcal{N}(0, R_t)
\]

where:

- \( R_t \in \mathbb{R}^{m \times m} \) is the observation noise covariance matrix
- Diagonal entries correspond to the variance of each sensor’s noise
- Larger entries in \(R_t\) mean we trust the sensor less  
- Smaller entries in \(R_t\) mean we trust the sensor more

---

## Encoder Ticks and Quantisation

Encoders measure angles in discrete steps called ticks.  
If an encoder has \( N_{\text{ticks}} \) ticks per revolution, the smallest measurable step is:

\[
\Delta_{\text{tick}} = \frac{2\pi}{N_{\text{ticks}}}
\]

A true joint angle \(\theta_{\text{true}}\) is quantised by rounding to the nearest tick:

\[
\theta_{\text{meas}} = \text{round}\left( \frac{\theta_{\text{true}}}{\Delta_{\text{tick}}} \right) \cdot \Delta_{\text{tick}}
\]

Quantisation introduces a non-Gaussian measurement error that is bounded within:

\[
-\frac{\Delta_{\text{tick}}}{2} \leq \theta_{\text{error}} \leq \frac{\Delta_{\text{tick}}}{2}
\]

---

## Measurement Dropout

Sometimes a sensor fails to report a value - for example, if the signal gets interruped.  

We can model this by a Bernoulli random variable:

\[
z_t \sim \text{Bernoulli}(p)
\]

where \(p\) is the probability of dropout.

- If \(z_t = 1\), the measurement is missing (often represented as NaN).  
- If \(z_t = 0\), the measurement is available.


Let \( Y_N \) be a r.v. representing the number of dropouts in \(N\) samples. 
- \( \mathbb{E} [ Y_N ] = N p \)
- \( \mathrm{Var} [Y_N] = N p (1-p) \)


---

## Bias Drift

Some sensors, like gyros, have a bias, \( b_t \) that slowly wanders over time. We can model this like a Brownian random walk:

\[
b_{t+1} = b_t + \eta_t
\]

Where:

\[
\eta_t \sim \mathcal{N}(0, \sigma_{\text{drift}}^2)
\]

Here, \(\sigma_{\text{drift}}\) is called the drift process noise standard deviation. 

---

### Sensor Noise (sensors/joint_models.py)

#### Encoder

This program simulates a noisy encoder and gyrometer for the angle and angular velocity, respectively, of a servo joint in a 2D 3DOF robot arm. 

For the noisy encoder, the goal is to simulate a measurement reading from a true data source. It takes in a the dropout probability, and observation noise variance. 

An encoder measures the joint angle in radians. 

A noisy encoder can be expressed mathematically as: 
\[ 
y_t = x_t + v_t 
\]

Where: \( v_t \sim \mathcal{N} (0, \sigma_{enc}^2) \)

#### Gyrometer

A gyrometer measures the angular velocity at the joint angle. This is the time derivative of the encoder measurements.  

A noisy gyrometer typically has bias drift (slowly wandering offset) and Gaussian measurement noise:

\[
y^{gyro}_t = \dot{\theta}_t + b_t + v^{gyro}_t
\]

Where the white measurement noise is:
\[
v^{gyro}_t \sim \mathcal{N}(0, \sigma_{gyro,t}^2)
\]

#### Bias random walk model (Brownian motion)

The bias is modelled as a discrete-time random walk:
\[
b_{t+1} = b_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, q_{b,t})
\]

The bias variance recursion (per discrete step) is:
\[
\mathrm{Var}[b_{t+1}] = \mathrm{Var}[b_t] + q_{b,t}
\]

This means bias variance grows over time because it accumulates drift.

#### Continuous-time Noise Density

If the sensor noise is in continuous-time density units, with:
- Encoder step \(\Delta t\)
- Gyro step \(\Delta t_g\)

Random-walk bias increment variance per step:
\[
q_{b,t} = \sigma_{drift, per s}^{2} \, \Delta t_g
\]

White-noise gyro variance per step:
\[
\sigma_{gyro,t}^{2} = \frac{\sigma_{gyro, density}^{2}}{\Delta t_g}
\]

Where:
- \(\sigma_{drift, per s}\) is the bias drift standard deviation density in \( \mathrm{rad/s}/\sqrt{\mathrm{s}} \)
- \(\sigma_{gyro, density}\) is the gyro white-noise density in \( \mathrm{rad/s}/\sqrt{\mathrm{Hz}} \) 

#### Per-sample Specification

But if you already know the per-sample noise values directly, then simply use:
\[
q_{b,t} = \sigma_{drift, step}^{2}
\]
\[
\sigma_{gyro,t}^{2} = \sigma_{gyro, step}^{2}
\]

#### Measurement Error

The total noise relative to the true angular velocity is:
\[
e_t = y^{gyro}_t - \dot{\theta}_t = b_t + v_t = b_{t-1} + \eta_t + v_t
\]

Assume \(v_t\) and \(\eta_t\) are independent of each other and of \(\dot{\theta}_t\).  

Conditional distribution given the previous bias:
\[
e_t \mid b_{t-1} \sim \mathcal{N}(b_{t-1}, \sigma_{gyro,t}^{2} + q_{b,t})
\]

Let \(\mathrm{Var}[b_{t-1}] = \sigma_{b,t-1}^{2}\)

Then the unconditional moments are:
\[
\mathbb{E}[e_t] = \mathbb{E}[b_{t-1}]
\]
\[
\mathrm{Var}[e_t] = \sigma_{b,t-1}^{2} + q_{b,t} + \sigma_{gyro,t}^{2}
\]

#### Multi-rate Sensors
If the gyro runs \(r\) times faster than the encoder \(( \Delta t_g = \Delta t / r )\), apply the above on the gyro timeline:
\[
t_k = k \, \Delta t_g
\]

Thus we interpolate the ground-truth angular velocity \(\dot{\theta}(t)\) to \(t_k\) before adding bias and noise.

## Kalman Filter Algorithm

#### Predict step (from process model)

State transition model:
\[
\hat{x}_{t+1|t} = A\,\hat{x}_{t|t}
\]
where:
- \(A\) is the state transition matrix, maps state from time \(t\) to \(t+1\) assuming no process noise.

Prediction step covariance:
\[
P_{t+1|t} = A P_{t|t} A^\top + Q
\]
where:
- \(Q\) is the process noise covariance matrix, models uncertainty added each step due to unmodelled dynamics or disturbances.

---

#### Update step (from measurement model)

Kalman Gain:
\[
K_t = P_{t|t-1} H^\top \left( H P_{t|t-1} H^\top + R \right)^{-1}
\]
where:
- \(H\) is the measurement matrix, maps state variables into measurement space.

Updated estimate:
\[
\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \left( y_t - H \hat{x}_{t|t-1} \right)
\]

Estimate covariance:
\[
P_{t|t} = (I - K_t H) P_{t|t-1}
\]

#### Constant Velocity State Transition

Without gyro bias: 
\[
x_t =
\begin{bmatrix}
\theta_t \\
\dot{\theta}_t
\end{bmatrix},
\quad
A =
\begin{bmatrix}
1 & \Delta t \\
0 & 1
\end{bmatrix}
\]

With gyro bias: 
\[
\tilde{x}_t =
\begin{bmatrix}
\theta_t \\
\dot{\theta}_t \\
b_t
\end{bmatrix},
\quad
\tilde{A} =
\begin{bmatrix}
1 & \Delta t & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\]

---

#### Statistical models

Encoder: 
\[
H_{enc} = [\,1 \quad 0\,], \quad
R_{enc} = [\,\sigma_{enc}^2\,]
\]

Gyro (no bias): 
\[
H_{gyro} = [\,0 \quad 1\,], \quad
R_{gyro} = [\,\sigma_{gyro}^2\,]
\]

- Per-step noise from continuous-time densities (gyro step \(\Delta t_g\)):
\[
\sigma_{gyro}^{2} = \frac{\sigma_{gyro,\ density}^{2}}{\Delta t_g}
\]

Gyro (with bias): 
\[
H_{gyro} = [\,0 \quad 1 \quad 1\,], \quad
q_b = \sigma_{drift}^2
\]

- Per-step noise from continuous-time densities (gyro step \(\Delta t_g\)):
\[
q_b = \sigma_{drift,\ density}^{2} \, \Delta t_g
\]

Process noise covariance: 
- Without bias: 
\[
Q = \mathrm{diag}(q_\theta,\ q_{\dot{\theta}})
\]
- With bias: 
\[
Q = \mathrm{diag}(q_\theta,\ q_{\dot{\theta}},\ q_b)
\]

# Dynamics

Dynamics describes how forces and torques generate motion in a robotic system. While kinematics focuses on positions and velocities, dynamics connects these to the physical forces required to move the robot’s joints and links.  

The equations of motion for an n-joint manipulator are typically expressed as:

\[
\boldsymbol{\tau} = M(\boldsymbol{q}) \, \ddot{\boldsymbol{q}} + C(\boldsymbol{q}, \dot{\boldsymbol{q}}) \, \dot{\boldsymbol{q}} + g(\boldsymbol{q})
\]

Where:

- \( \boldsymbol{q} \): joint angles (positions)
- \( \dot{\boldsymbol{q}} \): joint velocities
- \( \boldsymbol{\tau} \): joint torques  
- \( M(\boldsymbol{q}) \): joint-space inertia matrix  
- \( C(\boldsymbol{q}, \dot{\boldsymbol{q}}) \): Coriolis and centrifugal terms  
- \( g(\boldsymbol{q}) \): gravity vector  

This formulation allows us to simulate, control, and optimise robot motion under realistic physical constraints.

---

## Joint Torque

Torque, \( \tau \), represents the rotational force required at each joint to achieve a desired acceleration, and account for the coriolis effect, centrifugal force, and gravitional effect. 

---

## Coriolis Force

The Coriolis force arises when multiple joints of a robot move simultaneously, causing coupling between their velocities. 

Physically, it represents the additional torque needed to account for how one joint’s motion affects another due to changing coordinate orientations.  

Mathematically, the Coriolis terms appear in the term \( C(\boldsymbol{q}, \dot{\boldsymbol{q}}) \), which depends on both joint positions and velocities. 

These forces are velocity-dependent and always act perpendicular to the direction of motion, influencing how smoothly and accurately the robot can follow a trajectory. 

---

## Centrifugal Force

The centrifugal force is a component of the dynamic effects that arise when a rotating link experiences an outward “pull” due to its own angular velocity.  

In a robot manipulator, it represents the apparent force pushing mass away from the axis of rotation, requiring joint torques to counterbalance it.  

It is also contained within the \( C(\boldsymbol{q}, \dot{\boldsymbol{q}}) \) term, often combined with Coriolis effects.  

While Coriolis forces result from interactions between joints, centrifugal forces arise from the self-rotation of a single link. 

---

## Gravity

The gravity term represents the torque required at each joint to counteract the weight of the robot’s links when operating in a gravitational field. 

It depends only on the joint configuration \( \boldsymbol{q} \), since the direction of gravity is constant, and is independent of joint velocities or accelerations.

Each link \( i \) experiences a downward gravitational force:

\[
\mathbf{F}_{g,i} =
\begin{bmatrix}
0 \\ -m_i g
\end{bmatrix}
\]

where \( m_i \) is the link mass and \( g \) is the gravitational acceleration magnitude.

This force generates a moment (torque) about each upstream joint, depending on the link geometry and orientation.  

The total gravity torque vector is therefore:

\[
\boldsymbol{\tau}_g(\boldsymbol{q}) =
\sum_{i=1}^{n} J_{v,i}^T(\boldsymbol{q}) \, m_i \, \mathbf{g}
\]

where:

- \( J_{v,i}(\boldsymbol{q}) \) is the linear part of the Jacobian for link \( i \)’s centre of mass  (COM)
- \( \mathbf{g} = [0, -g, 0]^T \) is the gravitational acceleration vector  
- \( m_i \) is the mass of link \( i \)

This term expresses the torque needed to hold the robot still against gravity. When the robot is static (\( \dot{\boldsymbol{q}} = 0, \ddot{\boldsymbol{q}} = 0 \)), the equations of motion reduce to:

\[
\boldsymbol{\tau} = \boldsymbol{\tau}_g(\boldsymbol{q})
\]

Which defines the equilibrium torques required at each joint.

---

## Inertial Parameters

Inertial parameters describe how mass is distributed within each robot link.  
They define how the link resists both linear and angular acceleration.

For each link \(i\), the main parameters are:

- \( m_i \) : mass of the link  
- \( \mathbf{r}_{c,i} \) : position of the centre of mass (COM) in the link’s local frame  
- \( I_{zz,i} \) : rotational inertia about the z-axis through the COM  

---

## Inertia Tensor

The inertia tensor describes how a rigid body resists angular acceleration about its axes of rotation.  
It generalises the concept of scalar moment of inertia into a 3×3 matrix that captures how mass is distributed in space relative to a chosen reference frame.

Mathematically, for a rigid body with mass density \( \rho(\mathbf{r}) \), the inertia tensor about a point \( O \) is defined as:

\[
\mathbf{I}_O = 
\int_V \rho(\mathbf{r}) 
\begin{bmatrix}
y^2 + z^2 & -xy & -xz \\
-xy & x^2 + z^2 & -yz \\
-xz & -yz & x^2 + y^2
\end{bmatrix}
dV
\]

where \( (x, y, z) \) are the coordinates of each mass element relative to the point \( O \), and \( V \) is the body’s volume.

The inertia tensor is symmetric and positive definite, meaning it has real, positive eigenvalues corresponding to the body’s principal moments of inertia. 

When expressed about the centre of mass, it captures the pure rotational inertia of the body, and when shifted to other points (e.g. a joint), and the parallel-axis theorem is used to adjust it:

\[
\mathbf{I}_O = \mathbf{I}_C + m [\mathbf{r}_{OC}]_\times [\mathbf{r}_{OC}]_\times^T
\]

Where \( \mathbf{I}_C \) is the inertia about the COM and \( [\mathbf{r}_{OC}]_\times \) is the skew-symmetric matrix of the vector from the COM to the point \( O \).  

In robotics, this tensor is essential for computing the link’s contribution to the manipulator’s overall inertia matrix \( M(\boldsymbol{q}) \).

### Rotational Inertia of a Uniform Rod

#### Claim: For a uniform rod of length \( L_i \): \( I_{zz,i} = \frac{1}{12} m_i L_i^2 \)

#### Proof: 

Let the link \( i \) be a slender, uniform rod of total mass \( m_i \), parameterised by the coordinate \( x \in [0, L_i] \) along its local \( x \)-axis, with negligible thickness, so \( y = 0 \)

The linear mass density is constant:

\[
\lambda = \frac{m_i}{L_i}
\]

The COM for a uniform link lies at half the link length along its x-axis:

\[
\mathbf{r}_{c,i} =
\begin{bmatrix}
L_i / 2 \\ 0
\end{bmatrix}
\]

The moment of inertia about the \(z\)-axis through the base is given by: 

\[
I^{(0)}_{zz,i} = \int (x^2 + y^2)\, \mathrm{d}m 
\]

Placing the origin at the COM and integrating symmetrically over \( x \in [-L_i/2,\, L_i/2] \):

\[
I_{zz,i}
= \int_{-L_i/2}^{L_i/2} x^2\, \lambda\, \mathrm{d}x
= \lambda \left[ \frac{x^3}{3} \right]_{-L_i/2}^{L_i/2}
= \frac{2 \lambda}{3} \left( \frac{L_i}{2} \right)^3
= \frac{1}{12} m_i L_i^2
\]

---

## Spatial Motion Vector

A spatial motion vector combines angular and linear velocity into one vector:

\[
\mathbf{V} =
\begin{bmatrix}
\boldsymbol{\omega} \\
\mathbf{v}
\end{bmatrix}
\in \mathbb{R}^6
\]

- \( \boldsymbol{\omega} \): angular velocity  
- \( \mathbf{v} \): linear velocity of a point on the body  

This is the quantity that both \( \mathbf{S} \) and \( \mathbf{X} \) act upon.

---

## Screw Transforms \( \mathbf{S} \)

A screw is a compact representation of a rigid-body motion combining rotation and translation along an axis. 

In dynamics, each joint’s motion is defined by a screw axis \( \mathbf{S}_i \) describing the  direction and form of motion (rotation or translation) occurs in the joint’s local frame. 

Given a joint rate \( \dot{q}_i \), the spatial velocity contributed by that joint is:

\[
\mathbf{V}_i = \mathbf{S}_i \, \dot{q}_i
\]

In spatial vector dynamics, each joint’s motion is described by a 6×1 screw axis vector \( \mathbf{S} \). This vector captures both the rotational and translational components of motion in one unified form.

For joint \( i \) it's defined as:

\[
\mathbf{S}_i =
\begin{bmatrix}
\boldsymbol{\omega} \\
\mathbf{v}
\end{bmatrix}
\]

Where:

- \( \boldsymbol{\omega} \in \mathbb{R}^3 \): the angular velocity part.  
- \( \mathbf{v} \in \mathbb{R}^3 \): the linear velocity part.

For example, a revolute joint about the z-axis:

\[
\mathbf{S}_i =
\begin{bmatrix}
0 \\ 0 \\ 1 \\ 0 \\ 0 \\ 0
\end{bmatrix}
\]

For example, a prismatic joint along the x-axis:

\[
\mathbf{S}_i =
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 1 \\ 0 \\ 0
\end{bmatrix}
\]

The screw axis defines the subspace of motion the joint permits — its “allowed direction” in 6D space. Using screw theory ensures that spatial velocity and torque transformations remain coordinate-invariant and physically consistent.

---

## Spatial Transforms \( \mathbf{X} \)

The spatial transform \( \mathbf{X}_{j \leftarrow i} \) maps spatial motion vectors \( \mathbf{S} \) from frame \( i \) to frame \( j \):

\[
\mathbf{S}_j = \mathbf{X}_{j \leftarrow i} \, \mathbf{S}_i
\]

It encodes both the rotation and translation between the two frames, and always takes the 6×6 block form:

\[
\mathbf{X} =
\begin{bmatrix}
R & 0 \\
[r]_\times R & R
\end{bmatrix}
\]

Where:
- \( R \) is a 3×3 rotation matrix 
- \( [r]_\times \) is the skew-symmetric matrix of the translation vector \( r \)

---

## Spatial Inertia

Spatial inertia, \( I_s \), expresses both linear and angular inertia in a single 6×6 matrix, following the spatial vector formulation from Featherstone. 

For a rigid body with mass \( m \), COM position \( \mathbf{c} \), and inertia tensor \( I_C \) about the COM \( \mathbf{c} \):

\[
\mathbf{I}_s =
\begin{bmatrix}
I_o & m [\mathbf{c}]_\times \\ - m [\mathbf{c}]_\times & m I_3
\end{bmatrix}
\]

Where \( [\mathbf{c}]_\times \) is the skew-symmetric matrix of \( \mathbf{c} \):

\[
[\mathbf{c}]_\times =
\begin{bmatrix}
0 & -c_z & c_y \\
c_z & 0 & -c_x \\
-c_y & c_x & 0
\end{bmatrix}
\]

This representation allows for unified treatment of rotational and translational motion in the same coordinate frame. 

For a planar robot, we can embed each link into 3D space (with all motion about z-axis) and use spatial algebra for 3D dynamics.

Two fundamental spatial transforms are:

### Rotation about z-axis

\[
\mathbf{X}_{\text{rotZ}}(\theta) =
\begin{bmatrix}
R_z(\theta) & 0 \\
0 & R_z(\theta)
\end{bmatrix}
\]

With

\[
R_z(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\]

### Translation by vector \( \mathbf{r} = [r_x, r_y, r_z]^T \)

\[
\mathbf{X}_{\text{trans}}(\mathbf{r}) =
\begin{bmatrix}
I_3 & 0 \\
[\mathbf{r}]_\times & I_3
\end{bmatrix}
\]

These transforms allow motion and force quantities to be propagated along the kinematic chain.

---

