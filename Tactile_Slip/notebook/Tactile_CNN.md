# Incipient Slip Detection CNN

This project implements a Convolutional Neural Network (CNN) entirely from first principles using NumPy (without PyTorch or TensorFlow.)

The task is incipient slip classification using Bristol’s TacTip tactile dataset, which provides frame-by-frame marker displacements of the TacTip sensor pins during object contact.

---

## Problem Definition

#### Goal: Classify whether the tactile sensor is detecting object slip at each frame. 

Each input sample is a single frame rasterised into a 3-channel image:
1. Previous frame
2. Current frame
3. Absolute difference (motion magnitude)

The target label is binary:
- \( y = 1 \) is slip  
- \( y = 0 \) is stable contact  

---

## Dataset Pre-Processing Pipeline

The TacTip data are MATLAB `.mat` files containing 2D pin coordinates over time:
\[
\text{shape: } (T, 2M) \quad \text{where } M = \text{number of pins}
\]

### Step 1 – Normalisation
Each sequence is normalised so that pin coordinates are zero-mean and unit-variance:

\[
x' = \frac{x - \bar{x}}{\sigma} , 
\quad 
y' = \frac{y - \bar{y}}{\sigma}
\]

### Step 2 – Bilinear Rasterisation
Each frame’s pin coordinates are bilinearly splatted into a grayscale image of size \(H \times W\):

\[
I(u,v) = \sum_{k} w_k \; \delta(u - u_k, v - v_k)
\]

Where \(w_k\) are bilinear weights from fractional pixel offsets.

This yields smooth spatial activation maps preserving pin topology.

### Step 3 – Channel Stacking
For consecutive frames \(t{-}1\) and \(t\):

\[
X_t = [I_{t-1},\; I_t,\; |I_t - I_{t-1}|]
\]

Thus each sample \(X_t \in \mathbb{R}^{3\times H\times W}\)

### Step 4 – Slip Label Assignment
Pin displacements are used to define slip magnitude:
\[
v_t = \frac{1}{M} \sum_{i=1}^{M} \sqrt{(x_{t,i}-x_{t-1,i})^2+(y_{t,i}-y_{t-1,i})^2}
\]

A global slip threshold is estimated from the median and median-absolute-deviation (MAD) across all sequences:
\[
v_\text{thresh} = \text{median}(v) + 3\,\text{MAD}(v)
\]

Finally:
\[
y_t =
\begin{cases}
1 & \text{if } v_t \ge v_\text{thresh}\\
0 & \text{otherwise}
\end{cases}
\]

### Step 5 – Normalisation Across Dataset
After all frames are rasterised, dataset-level mean and std are computed:
\[
X_{\text{norm}} = \frac{X - \mu_X}{\sigma_X}
\]

---

## Model Architecture


| Layer | Type | Input | Output | Activation |
|--------|------|--------|---------|-------------|
| 1 | Conv3D | (3, 96, 96) | (8, 96, 96) | ReLU |
| 2 | MaxPool3D | (8, 96, 96) | (8, 48, 48) | — |
| 3 | Conv3D | (8, 48, 48) | (8, 48, 48) | ReLU |
| 4 | MaxPool3D | (8, 48, 48) | (8, 24, 24) | — |
| 5 | Flatten | (8, 24, 24) | (4608,) | — |
| 6 | Linear | (4608,) | (1,) | Sigmoid |

Each layer’s output dimensions are computed by:
\[
H_\text{out} = \left\lfloor\frac{H_\text{in} + 2p - k}{s}\right\rfloor + 1
\]

---

## Mathematical Formulation of Each Layer

### 1. Convolution
For kernel \(W_c \in \mathbb{R}^{C_\text{in}\times k\times k}\) and bias \(b_c\):

\[
z_c(i,j) = \sum_{d=1}^{C_\text{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} 
W_{c,d,u,v} \; x_d(i+u, j+v) + b_c
\]

### 2. ReLU Activation
\[
a_c(i,j) = \max(0,\, z_c(i,j))
\]

### 3. Max Pooling
Over local window size \(s \times s\):
\[
p_c(i,j) = \max_{u,v < s} a_c(i s + u, j s + v)
\]

### 4. Flatten
\[
\text{flatten}(x) : \mathbb{R}^{C\times H\times W} \to \mathbb{R}^{CHW}
\]

### 5. Linear Layer
\[
z = Wv + b
\]

### 6. Sigmoid
\[
p = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

### 7. Binary Cross-Entropy (BCE) Loss

For true label \(y \in \{0,1\}\) and predicted probability \(p\):

\[
L = -\big[y\log p + (1-y)\log(1-p)\big]
\]

For numerical stability, the loss is computed directly from logits \(z\):

\[
L = (1-y)\, \mathrm{softplus}(z) + y\, \mathrm{softplus}(-z)
\]

Where \(\mathrm{softplus}(z)=\log(1+e^z)\)

---

## Backpropagation

Every operation has a corresponding backward pass manually derived.

### BCE + Sigmoid Backward
From \(p=\sigma(z)\):
\[
\frac{\partial L}{\partial z} = p - y
\]

### Linear Backward
Given \(z = W x + b\):

\[
\frac{\partial L}{\partial W} = (\frac{\partial L}{\partial z}) x^\top, \quad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}, \quad
\frac{\partial L}{\partial x} = W^\top (\frac{\partial L}{\partial z})
\]

### ReLU Backward
\[
\frac{\partial L}{\partial x} = 
\begin{cases}
\frac{\partial L}{\partial y} & \text{if } x>0\\
0 & \text{else}
\end{cases}
\]

### MaxPool Backward
Gradient flows only to the maximal element within each pooling window.

### Convolution Backward
\[
\frac{\partial L}{\partial W_{c}} 
= \sum_{i,j} \frac{\partial L}{\partial z_{c}(i,j)}\, x[i:i+k, j:j+k]
\]
\[
\frac{\partial L}{\partial x} = W_{c} * \frac{\partial L}{\partial z_{c}}
\]

---

## Parameter Initialisation

To ensure proper variance propagation:

- **Convolution layers:** He normalisation  
  \[
  W \sim \mathcal{N}(0, \sqrt{\tfrac{2}{\text{fan\_in}}})
  \]
- **Linear layer:** Xavier initialisation  
  \[
  W \sim \mathcal{N}\!\left(0, \sqrt{\tfrac{2}{\text{fan\_in}+\text{fan\_out}}}\right)
  \]

Biases are zero-initialised.

---

## Training Procedure

### Forward Pass
Each sample is propagated through all layers to compute logits \(z\) and probability \(p\).

### Loss and Metrics
Binary Cross-Entropy loss is computed per sample.

Precision, recall, F1, and accuracy are evaluated from sigmoid probabilities.

### Backward Pass
Gradients are propagated in reverse order:
\[
\text{Linear} \leftarrow \text{Flatten} \leftarrow \text{MaxPool} \leftarrow \text{ReLU} \leftarrow \text{Conv}
\]

### Optimisation
Parameters are updated with stochastic gradient descent (SGD):
\[
\theta \leftarrow \theta - \eta\,(\nabla_\theta L + \lambda \theta)
\]

Where \(\lambda\) is a weight-decay coefficient. 

### Balanced Mini-Batching
To handle class imbalance (only ~2% positive examples), each batch samples equal numbers of slip and non-slip frames.

---

## Evaluation

After training:
- Collect logits on train/val/test splits
- Sweep thresholds \(t\in[0.01,0.99]\)
- Choose threshold \(t^*\) that maximises F1-score on validation
- Compute final metrics on test at \(t=t^*\)

### Metrics Computed
\[
\text{Precision} = \frac{TP}{TP+FP},\quad
\text{Recall} = \frac{TP}{TP+FN}
\]
\[
F_1 = \frac{2PR}{P+R},\quad
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
\]

---

## Main Modules

| File | Purpose |
|------|----------|
| `functional_numpy.py` | Forward-pass operators (conv, pool, relu, linear, sigmoid) |
| `functional_numpy_backward.py` | Manual gradient definitions |
| `cnn_numpy.py` | CNN forward model and caching of activations |
| `train_cnn_numpy.py` | Training loop, He/Xavier init, SGD updates |
| `metrics_numpy.py` | Precision/recall/F1 metrics and threshold sweep |
| `rasterise_numpy.py` | TacTip pre-processing, rasterisation, labelling |
| `preprocess_tactip_numpy.py` | Full data pipeline to `.npz` files |
| `train_tactip_numpy.py` | Training script |
| `eval_tactip_numpy.py` | Evaluation script and threshold tuning |
| `experiment_numpy.py` | Utility: batching, config loading, logging |

---

## Summary

1. **Low-level mechanics of CNNs:** Understood convolution, pooling, and nonlinear activations at array-index level.

2. **Manual backpropagation:** Derived and coded gradients for all layers without autograd.

3. **Loss and optimisation from first principles:** Implemented stable BCE with logits and SGD with weight decay. 

4. **Tactile data rasterisation:** Designed bilinear “pin-splatting” into image space to convert point clouds to CNN inputs. 

5. **Metrics and threshold calibration:** Implemented full F1-optimised binary classification pipeline.

6. **End-to-end NumPy learning system:** From tactile `.mat` files, to image tensors, into CNN, then evaluation.
