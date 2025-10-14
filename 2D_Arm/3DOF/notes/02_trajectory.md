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
