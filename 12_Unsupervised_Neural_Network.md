# 12.0 Self-Organized Learning
- [i] Self-Organized Learning = Unsupervised Learning
- No external teacher, i.e., no manual labels.

- [i] Training process of self-organized learning:
- Receive a number of different input patterns;
- Discover significant features from these patterns;
- Learn how to classify input data.

# 12.1 Hebbian Learning
## 12.1.1 Hebb's Law
- [i] Hebb's Law describes that:
- If:
	- A neuron $i$ is near enough to excite neruon $j$,
	- and it repeatedly participates in its activation;
- Then:
	- The synaptic connection between the two neurons is strengthened;
	- Neuron $j$ becomes more sensitive to stimuli from neuron $i$.

- [i] Hebb's Law's description in *two rules:*
- [I] Rule 1:
	- If:
		- The two neurons on either side of a connection are activated *synchronously;*
	- Then:
		- The weight of that connection is *increased*.
- [I] Rule 2:
	- If:
		- The two neurons on either side of a connection are activated *asynchronously;*
	- Then:
		- The weight of that connection is *decreased*.

## 12.1.2 Weight Adjustments
![[Hebbian_Activation.png]]
- [i] Weight adjustments according to the Hebb's Law could be described as follows.
$$
\Delta w_{ij}(p) = F\Bigl(y_i(p), x_i(p)\Bigr)
$$
This formula regulates that the weight connecting from the neuron $i$ to the neuron $j$ is determined by:
- The input to the neuron $i$;
- The output from the neuron $i$;
where $F$ is an arbitrary customed function.

- [i] The **activity product rule** specialize the above formula by regulating the function with:
$$
F(v_1,v_2)=\alpha \cdot v_1 \cdot v_2
$$
Applying this to the original rule:
$$
\Delta w_{ij}(p)=\alpha\cdot y_j(p)\cdot x_j(p)
$$
where $\alpha$ is the *learning rate* parameter.

- [*] To prevent the monotonical increase of weights, we introduce a *non-linear forgetting factor* to the Hebb's Law:
$$
\Delta w_{ij}(p)=\alpha\cdot y_j(p)\cdot x_j(p)-\phi\cdot y_j(p)\cdot w_{ij}(p)
$$
where $\phi$ is the forgetting factor.

## 12.1.3 Hebbian Learning Algorithm
![[Hebbian_Network.png|200]]
### Step 1. Initialization
#### Step 1.1 Initialize weights and biases
Suppose that number of neurons is $M$, number of input is $N$.

$$
\begin{align}
\mathbf{W}^{(0)} &= \begin{bmatrix}
- & \mathbf{w}_{01}^\top & - \\
- & \mathbf{w}_{02}^\top & - \\
&\vdots& \\
- & \mathbf{w}_{0M}^\top & - \\
\end{bmatrix}, \ \mathbf{w}_{0i}\in\mathbb{R}^N
\\ \\
&=\begin{bmatrix}
w_{01,1} & w_{01,2} & \cdots & w_{01,N} \\
w_{02,1} & w_{02,2} & \cdots & w_{02,N} \\
\vdots & \vdots & \ddots & \vdots \\
w_{0M,1} & w_{0M,2} & \cdots & w_{0M,N}
\end{bmatrix}\in\mathbb{R}^M\times\mathbb{R}^N
\\ \\
\mathbf{b} &= \begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ b_N
\end{bmatrix}
\end{align}
$$
#### Step 1.2 Hyper-Patameters
- Learning Rate: $\eta$
- Decay (Forget) Factor: $\phi$
### Step 2. Iteration: For all $\mathbf{x}_i$
#### Step 2.1 Activation
The output is given by:
$$
y_{i}^{(t)} = f(\mathbf{W}^{(t)}\mathbf{x}_i-\mathbf{b})\in\mathbb{R}^M
$$
where $f$ is the activation function.
#### Step 2.2 Learning
The  change in weights:
$$
\begin{align}
\Delta\mathbf{W}^{(t)} &= \eta\cdot(y_{i}^{(t)}\cdot\mathbf{x}_i^\top)-\phi\cdot\mathbf{W}^{(t)} 
\\ \\
\mathbf{W}^{(t+1)} &= \mathbf{W}^{(t)}+\Delta\mathbf{W}^{(t)}
\end{align}
$$
where $\eta$ is the learning rate, $\phi$ is the decay factor.
Note that:
$$
\Delta\mathbf{W}^{(t)}\in\mathbb{R}^{M}\times\mathbb{R}^{N}
$$

## 12.1.4 Hebbian Learning Example
**Given:**
- A fully connected feed-forward network.
	- Single layer, $5$ computation neurons.
- Learning rate: $\eta=0.1$
- Training inputs:
$$
\mathbf{x}_1=\begin{bmatrix}
0 \\ 0 \\ 0 \\ 0 \\ 0
\end{bmatrix},
\
\mathbf{x}_2=\begin{bmatrix}
0 \\ 1 \\ 0 \\ 0 \\ 1
\end{bmatrix},
\
\mathbf{x}_3=\begin{bmatrix}
0 \\ 0 \\ 0 \\ 1 \\ 0
\end{bmatrix},
\
\mathbf{x}_4=\begin{bmatrix}
0 \\ 0 \\ 1 \\ 0 \\ 0
\end{bmatrix},
\
\mathbf{x}_5=\begin{bmatrix}
0 \\ 1 \\ 0 \\ 0 \\ 1
\end{bmatrix}
$$

**Do:**
1. Initialize weights and Bias.
$$
\mathbf{W}_0=\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}, \ 
\mathbf{b}_0=\begin{bmatrix}
0.1 \\ 0.1 \\ 0.1 \\ 0.1 \\ 0.1
\end{bmatrix}
$$
---
*Iteration 1.*
Activation.
$$
\begin{align}
y_1 &= \mathbf{W}\mathbf{x}_1-b
\\ \\
&= \begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 0 \\ 0
\end{bmatrix} - 
\begin{bmatrix}
0.1 \\ 0.1 \\ 0.1 \\ 0.1 \\ 0.1
\end{bmatrix}
\\ \\
&=\begin{bmatrix}
-0.1 \\ -0.1 \\ -0.1 \\ -0.1 \\ -0.1
\end{bmatrix}
\end{align}
$$
 Learning
$$
\begin{align}
\Delta\mathbf{W}_0 &= \eta\cdot y_1\cdot\mathbf{x}_1^\top
\\ \\
&= 0.1 \cdot \begin{bmatrix}
-0.1 \\ -0.1 \\ -0.1 \\ -0.1 \\ -0.1
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 0 & 0 & 0
\end{bmatrix}
\\ \\
&= \begin{bmatrix}
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
\\ \\
\implies
\mathbf{W}_1 &= \mathbf{W}_0+\Delta\mathbf{W}_0
\\ \\
&=\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}, 
\end{align}
$$
----
*Iteration 2.*
Activation
$$
\begin{align}
y_2 &= \mathbf{W}_1\mathbf{x}_2-b
\\ \\
&= \begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
0 \\ 1 \\ 0 \\ 0 \\ 1
\end{bmatrix} - 
\begin{bmatrix}
0.1 \\ 0.1 \\ 0.1 \\ 0.1 \\ 0.1
\end{bmatrix}
\\ \\
&=\begin{bmatrix}
-0.1 \\ 0.9 \\ -0.1 \\ -0.1 \\ 0.9
\end{bmatrix}
\end{align}
$$
 Learning
$$
\begin{align}
\Delta\mathbf{W}_1 &= \eta\cdot y_2\cdot\mathbf{x}_2^\top
\\ \\
&= 0.1 \cdot \begin{bmatrix}
-0.1 \\ 0.9 \\ -0.1 \\ -0.1 \\ 0.9
\end{bmatrix}
\begin{bmatrix}
0 & 1 & 0 & 0 & 1
\end{bmatrix}
\\ \\
&= \begin{bmatrix}
0 & -0.01 & 0 & 0 & -0.01 \\
0 & 0.09 & 0 & 0 & 0.09 \\
0 & -0.01 & 0 & 0 & -0.01 \\
0 & -0.01 & 0 & 0 & -0.01 \\
0 & 0.09 & 0 & 0 & 0.09 \\
\end{bmatrix}
\\ \\
\implies
\mathbf{W}_2 &= \mathbf{W}_1+\Delta\mathbf{W}_1
\\ \\
&=\begin{bmatrix}
1 & -0.01 & 0 & 0 & -0.01 \\
0 & 1.09 & 0 & 0 & 0.09 \\
0 & -0.01 & 1 & 0 & -0.01 \\
0 & -0.01 & 0 & 1 & -0.01 \\
0 & 0.09 & 0 & 0 & 1.09 \\
\end{bmatrix}, 
\end{align}
$$
---
*More Iterations....*

# 12.2 Competitive Learning
## 12.2.1 Basics
- [i] Neurons compete among themselves to be activated.
	- Only a *single* output neuron is active at any time.
	- The output neuron that wins is the *winner-takes-all* neuron.

> Teuvo Kohonen: Self-organizing feature maps.

## 12.2.2 Kohonen Network
- [i] Structure of a Kohonen Network
	- Input: Fixed number of input;
	- Competition Layer: Higher dimensional Kohonen Layer.

## 12.2.3 Competitive Learning Algorithm
![[Kohonen_Network.png]]
### Step 1. Initialization
#### Step 1.1 Initialize Weights
Suppose we have an $P\times Q$ self-organizing map. We initialize the weights as:
$$
\mathcal{W}=\begin{bmatrix}
\mathbf{w}_{11} & \mathbf{w}_{12} & \cdots & \mathbf{w}_{1Q}
\\
\mathbf{w}_{21} & \mathbf{w}_{22} & \cdots & \mathbf{w}_{2Q}
\\
\vdots & \vdots & \ddots & \vdots
\\
\mathbf{w}_{P1} & \mathbf{w}_{P2} & \cdots & \mathbf{w}_{PQ}
\\
\end{bmatrix}\in\mathbb{R}^P\times\mathbb{R}^Q\times\mathbb{R}^N
$$
where
$$
\mathbf{w}_{ij}=\begin{bmatrix}
w_{ij,1} \\ w_{ij,2} \\ \vdots \\ w_{ij,N}
\end{bmatrix} \in\mathbb{R}^{N}
$$
is the matrix for the neuron at position $i,j$.

#### Step 1.2 Hyper-Parameters
- Learning rate: $\eta$,
- Neighbour radius: $r$.
### Step 2. Iteration: For all $\mathbf{x}_{t}$
#### Step 2.1 Calculate the winning vector
$$
(i^{(t)},j^{(t)})=\text{argmin}_{i,j}\|\mathbf{x}_{t}-\mathbf{w}_{ij}\|
$$
#### Step 2.2 Update weight of the winner
$$
\mathbf{w}_{i^{(t)}j^{(t)}}^{(t+1)}=\mathbf{w}_{i^{(t)}j^{(t)}}^{(t)}+\eta\cdot(\mathbf{x}-\mathbf{w}_{i^{(t)}j^{(t)}}^{(t)})
$$


