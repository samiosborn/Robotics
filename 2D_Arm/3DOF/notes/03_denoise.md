# Denoise

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
