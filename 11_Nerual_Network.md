# 11.1 A Neuron
- [i] The neuron is a simple computing unit. 
- The neuron computes the weighted sum of its input signals.
- Then it activates the input signals using an activation function.

## 11.1.1 Perceptron's Training Algorithm
### Step 1. Initialization
Set the initial weights $\mathbf{w}=\begin{bmatrix}w_{1}\\w_{2}\\\cdots\\w_{n}\end{bmatrix}$ and threshold $\theta$ to random numbers in the range of $[-0.5, 0.5]$.

### Step 2. Activation
Activate the perceptron by applying inputs $\mathbf{x}_{p}$ and desired output $Y_{dp}$ at iterations of $p=1,2,\cdots$.
$$
Y_{p}=\sigma\Bigl(\mathbf{w}_{p}^\top\mathbf{x}_{p}-\theta\Bigr)
$$
### Step 3. Weight Training
Calculate the error of the output.
$$
e_{p}=Y_{dp}-Y_{p}
$$
The weight correction is computed by the delta rule:
$$
\nabla \mathbf{w}_{p}=\alpha \cdot \mathbf{x}_{p}\cdot e_{p}
$$
Update the weights of the perceptron using the weight correction.
$$
\mathbf{w}_{p+1}=\mathbf{w}_{p}+\nabla \mathbf{w}_{p}
$$
# 11.2 Back-Propagation Neural Network
![[NN_Example.png]]
Forward Propagation:
$$
\begin{align}
y_{3} &= \sigma(x_{1}\cdot w_{13}+x_{2}\cdot w_{23}-\theta_{3}) \\
y_{4} &= \sigma(x_{1}\cdot w_{14}+x_{2}\cdot w_{24}-\theta_{4}) \\
y_{5} &= \sigma(y_{3}\cdot w_{35}+y_{4}\cdot w_{45}-\theta_{5})
\end{align}
$$
Backward Propagation:
$$
\begin{align}
\mathcal{L} &= \frac{1}{2}(y_{d}-y_{5})^2 \\
w_{35}^{(1)} &= w_{35}^{(0)}-\eta \cdot \dfrac{\partial \mathcal{L}}{\partial w_{35}} \\ \\
&= w_{35}^{(0)}-\eta \cdot \dfrac{\partial\mathcal{L}}{\partial y_{5}}\cdot\dfrac{\partial y_{5}}{\partial w_{35}} \\
&= w_{35}^{(0)}-\eta \cdot \Bigl(-(y_{d}-y_{5})\Bigr)\cdot \biggl(y_{3}\cdot\sigma'\Bigl(y_{3}\cdot w_{35}^{(0)}+y_{4}\cdot w_{45}^{(0)}-\theta_{5}^{(0)}\Bigr)\biggr) \\ \\
&= w_{35}^{(0)}-\eta \cdot \Bigl(-()\Bigr)
\end{align}
$$
$$
\begin{align} \\
\theta_{5}^{(1)} &= \theta_{35}^{(0)}-\eta \cdot \dfrac{\partial\mathcal{L}}{\partial w_{35}^{(0)}} \\
&= \theta_{35}^{(0)}-\eta \cdot\dfrac{\partial \mathcal{L}}{\partial y_{35}}\cdot\dfrac{\partial y_{35}}{\partial \theta_{5}} \\
&= \theta_{35}^{(0)} - \eta \cdot\Bigl(-(y_{d}-y_{5})\Bigr)\cdot \biggl(-\sigma'\Bigl(y_{3}\cdot w_{35}^{(0)}+y_{4}\cdot w_{45}^{(0)}-\theta_{5}^{(0)} \Bigr)\biggr)
\end{align}
$$


