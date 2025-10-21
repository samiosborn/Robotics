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

## Forces

### Joint Torque

Torque, \( \tau \), represents the rotational force required at each joint to achieve a desired acceleration, and account for the coriolis effect, centrifugal force, and gravitional effect. 

---

### Coriolis Force

The Coriolis force arises when multiple joints of a robot move simultaneously, causing coupling between their velocities. 

Physically, it represents the additional torque needed to account for how one joint’s motion affects another due to changing coordinate orientations.  

Mathematically, the Coriolis terms appear in the term \( C(\boldsymbol{q}, \dot{\boldsymbol{q}}) \), which depends on both joint positions and velocities. 

These forces are velocity-dependent and always act perpendicular to the direction of motion, influencing how smoothly and accurately the robot can follow a trajectory. 

---

### Centrifugal Force

The centrifugal force is a component of the dynamic effects that arise when a rotating link experiences an outward “pull” due to its own angular velocity.  

In a robot manipulator, it represents the apparent force pushing mass away from the axis of rotation, requiring joint torques to counterbalance it.  

It is also contained within the \( C(\boldsymbol{q}, \dot{\boldsymbol{q}}) \) term, often combined with Coriolis effects.  

While Coriolis forces result from interactions between joints, centrifugal forces arise from the self-rotation of a single link. 

---

### Gravity

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

## Lagrangian Dynamics Model 

### Claim:

For a manipulator with generalised coordinates \( \mathbf{q} \in \mathbb{R}^n \), the equations of motion can be written in the Lagrangian form:

\[
\boxed{
\mathbf{M}(\mathbf{q}) \, \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \, \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}
}
\]

---

### Proof:

We begin from first principles with Newton’s laws applied to a system with holonomic constraints (which constrain motion).

---

#### Kinetic and Potential energy

For a system of \( n \) generalised coordinates \( \mathbf{q} = [q_1, \ldots, q_n]^T \), each link \( i \) has translational and rotational kinetic energy, \( T_i \), due to linear and rotation movements, respectively, of it's centre of mass: 

\[
T_i = \frac{1}{2} m_i \, \dot{\mathbf{p}}_{c,i}^T \dot{\mathbf{p}}_{c,i} + \frac{1}{2} \boldsymbol{\omega}_i^T \mathbf{I}_i \boldsymbol{\omega}_i
\]

- \( \mathbf{I}_i \) represents the inertia tensor. 

Alongside gravitational potential energy, \( U_i \) :

\[
U_i = m_i g \, \mathbf{p}_{c,i}^T \hat{\mathbf{z}}
\]

The Lagrangian is: 

\[
L(\mathbf{q}, \dot{\mathbf{q}}) = \sum_{i=1}^n (T_i - U_i)
\]

---

#### Linear and Angular Jacobians

The position and angular velocity of each link are functions of the generalised coordinates:

\[
\mathbf{p}_{c,i} = \mathbf{p}_{c,i}(\mathbf{q})
\]

Thus, 

\[
\boldsymbol{\omega}_i = \mathbf{J}_{\omega,i}(\mathbf{q}) \, \dot{\mathbf{q}}
\]

\[
\dot{\mathbf{p}}_{c,i} = \mathbf{J}_{v,i}(\mathbf{q}) \, \dot{\mathbf{q}}
\]

Where \( \mathbf{J}_{v,i} \) and \( \mathbf{J}_{\omega,i} \) are the linear and angular velocity Jacobians.

---

#### Kinetic energy in matrix form

Substitute into \( T_i \):

\[
T_i = \frac{1}{2} \dot{\mathbf{q}}^T 
\left( 
m_i \mathbf{J}_{v,i}^T \mathbf{J}_{v,i} + \mathbf{J}_{\omega,i}^T \mathbf{I}_i \mathbf{J}_{\omega,i}
\right)
\dot{\mathbf{q}}
\]

Summing over all links gives the total kinetic energy

\[
T = \frac{1}{2} \dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}}
\]

Where we have the mass matrix: 

\[
\mathbf{M}(\mathbf{q}) = \sum_i 
\left(
m_i \mathbf{J}_{v,i}^T \mathbf{J}_{v,i} + \mathbf{J}_{\omega,i}^T \mathbf{I}_i \mathbf{J}_{\omega,i}
\right)
\]

- Both symmetric and positive definite.

---

#### Lagrange’s equation

For each coordinate \( q_j \), Lagrange’s equation states: 

\[
\frac{d}{dt}\left( \frac{\partial L}{\partial \dot{q}_j} \right) - \frac{\partial L}{\partial q_j} = \tau_j
\]

Since:
\[
L(\mathbf{q}, \dot{\mathbf{q}}) = \frac{1}{2} \dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}} - U(\mathbf{q})
\]


We have: 
\[
\frac{\partial L}{\partial \dot{\mathbf{q}}} = \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}}
\]

Then:
\[
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\mathbf{q}}}\right)
= \mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \dot{\mathbf{M}}(\mathbf{q}) \dot{\mathbf{q}}
\]

Also,

\[
\frac{\partial L}{\partial \mathbf{q}}
= \frac{1}{2} \dot{\mathbf{q}}^T \frac{\partial \mathbf{M}}{\partial \mathbf{q}} \dot{\mathbf{q}} - \frac{\partial U}{\partial \mathbf{q}}
\]

Substitute into Lagrange’s equation:

\[
\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \dot{\mathbf{M}}(\mathbf{q}) \dot{\mathbf{q}} - \frac{1}{2} \dot{\mathbf{q}}^T \frac{\partial \mathbf{M}}{\partial \mathbf{q}} \dot{\mathbf{q}} + \frac{\partial U}{\partial \mathbf{q}} = \boldsymbol{\tau}
\]

---

#### Coriolos and Centrifugal Matrix

Using Christoffel symbols of the first kind,

\[
c_{ijk} = \frac{1}{2}\left(
\frac{\partial M_{ij}}{\partial q_k} +
\frac{\partial M_{ik}}{\partial q_j} -
\frac{\partial M_{jk}}{\partial q_i}
\right)
\]

We define the Coriolis and centrifugal matrix \( \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \) by

\[
[\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}]_i = \sum_{j,k} c_{ijk} \dot{q}_j \dot{q}_k
\]

---

#### Gravitational Matrix
The gravitational term is:

\[
\mathbf{g}(\mathbf{q}) = \frac{\partial U}{\partial \mathbf{q}}
\]

Hence, the equations of motion become: 

\[
\mathbf{M}(\mathbf{q}) \, \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \, \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) 
= \boldsymbol{\tau}
\]

---

## Inertia

### Inertial Parameters

Inertial parameters describe how mass is distributed within each robot link.  
They define how the link resists both linear and angular acceleration.

For each link \(i\), the main parameters are:

- \( m_i \) : mass of the link  
- \( \mathbf{r}_{c,i} \) : position of the centre of mass (COM) in the link’s local frame  
- \( I_{zz,i} \) : rotational inertia about the z-axis through the COM  

---

### Inertia Tensor

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

#### Rotational Inertia of a Uniform Rod

##### Claim: For a uniform rod of length \( L_i \): \( I_{zz,i} = \frac{1}{12} m_i L_i^2 \)

##### Proof: 

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

## Spatial Dynamics

### Spatial Motion Vector

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

### Screw Transforms \( \mathbf{S} \)

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

### Spatial Transforms \( \mathbf{X} \)

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

### Spatial Inertia

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

#### Rotation about z-axis

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

#### Translation by vector \( \mathbf{r} = [r_x, r_y, r_z]^T \)

\[
\mathbf{X}_{\text{trans}}(\mathbf{r}) =
\begin{bmatrix}
I_3 & 0 \\
[\mathbf{r}]_\times & I_3
\end{bmatrix}
\]

These transforms allow motion and force quantities to be propagated along the kinematic chain.

---

