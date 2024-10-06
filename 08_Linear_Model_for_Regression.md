### Noting Paradigm
- $x$ - Plain text: Scalar.
- $\mathbf{x}$ - Bold-Face lowercase: Vector of scalars.
	- e.g., $\mathbf{x}=\begin{bmatrix}x_1 \\ x_2 \\ \cdots \\ x_D\end{bmatrix}$, where $\mathbf{x}\in \mathbb{R}^D$
- $\mathbf{X}$ - Bold-Face uppercase: Set of vectors.
	- e.g., $\mathbf{X}=\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_N\}$, where $\mathbf{X}\subset\mathbb{R}^D$ and $|\mathbf{X}|=N$.
# 8.0 Regression
## 8.0.0 Why Regression?
### Problem Setup
**Given**
- A set of inputs:
	- $\mathbf{X}=\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_N\}$, where, for each input:
		- $\mathbf{x}_i=\begin{bmatrix}x_{i1} \\ x_{i2} \\ \cdots \\ x_{iD}\end{bmatrix}\in\mathbb{R}^D$
- A set of corresponding outputs:
	- $\mathbf{y}=\{y_1,y_2,\cdots,y_N\}$, where, for each output:
		- $y_i\in\mathbb{R}$
- A labelling relation:
	- $D=\{(\mathbf{x}_1,y_1),(\mathbf{x}_2,y_2),\cdots,(\mathbf{x}_N,y_N)\}$
**Goal**
- To learn a mapping
	- $f(x): \mathbf{X}\rightarrow\mathbf{y}$, where
		- Input: An arbitrary vector $\mathbf{x}\notin\mathbf{X}$, 
		- Output: A scalar $y\notin\mathbf{y}$.
- Such that we could make prediction
	- about $y_*$
	- on an unseen input $\mathbf{x}_*$ is encountered.
### Different Kinds
- Parametric Regression 参数性回归
	- Assume a functional form for $f(x)$.
- Nonparametric Regression 非参数性回归
	- Does not assume a functional form for $f(x)$.
### To sum up
- From given relations, learn a function that
	- Takes a vector as an input
	- Output a real number that
		- "Fits" the given pattern
## 8.0.1 Definition

- [?] What is Regression?
- Regression aims at modelling the dependence of:
	- a response $Y$,
	- on a covariate $X$.
- That is, to predict the value of one or more continuous target variables $y$ given the value of input vector $x$.

- [i] The regression model is described by
$$
y=f(\mathbf{x})+\epsilon
$$
- where the dependence of a response $y$ on a covariate $\mathbf{x}$ is captured via:
	- $p(y|\mathbf{x})$,
	- i.e., a conditional probability distribution.

- [i] Conditional Mean of a regression function
- Considering the Mean Squared Error, we find the MMSE estimate:
	- $\mathcal{E}(f)=\mathbb{E}(y-f(\mathbf{x}))^2$
		- $=\int\idotsint (y-f(\mathbf{x}))^2 \cdot p(\mathbf{x},y) \ d\mathbf{x}dy$
		- $=\int\idotsint(y-f(\mathbf{x}))^2\cdot p(\mathbf{x})\cdot p(y|\mathbf{x}) d\mathbf{x}y$
		- $=\idotsint p(\mathbf{x})\cdot\int\biggl[\Bigl(y-f(\mathbf{x})\Bigr)^2\cdot p(y|\mathbf{x})dy\biggr]d\mathbf{x}$
		  (Every dimension is integrated)
	- Therefore, we need to minimize:
		- $\int \Bigl(y-f(x)\Bigr)^2\cdot p(y|\mathbf{x})dy$
			- $\implies \dfrac{\partial}{\partial f(\mathbf{x})}\int \biggl[\Bigl(y-f(x)\Bigr)^2\cdot p(y|\mathbf{x})dy\biggr]=0$
			- $\implies f(x)=\int y\cdot p(y|\mathbf{x}) \ dy=\mathbb{E}[y|\mathbf{x}]$
- That is to say,
	- $f(x)$ we wanted is the **Conditional Mean** of $y$ given covariate $\mathbf{x}$.
# 8.1 Linear Regression
## 8.1.0 Affine Function

- [?] (Additional) What is an affine function (仿射函数)?
- A function that:
	- Takes a vector input, and
	- Outputs a scalar.
	- i.e., $f: \mathbb{R}^N\mapsto\mathbb{R}$ is a general form of an affine function.
- More generally, an "affine transformation" (仿射变换) denotes:
	- $\mathbb{R}^n\mapsto\mathbb{R}^m$
		- Turning an $n$-d vector to an $m$-d one.
	- $\mathbf{x}\mapsto A\mathbf{x}+b$ is a more general description of an affine transformation, where
		- $A$ is an $m\times n$ matrix, and
		- $b$ is an $m$-d vector.
	- When $m=1$, the affine transformation denotes an **affine function**.
## 8.1.1 Definition
- [i] Linear Regression
- $f(x)$, that is the conditional mean, is an **affine function** of $\mathbf{x}$.
	- $f(x)=\Bigl[w_1\phi_1(\mathbf{x})+w_2\phi_2(\mathbf{x})+\cdots+w_M\phi_M(\mathbf{x})\Bigr]+w_0\phi_0(\mathbf{x})$
		- $=\sum_{j=1}^{M}w_j\phi_j(\mathbf{x})+w_0\phi_0(\mathbf{x})$
		- $\begin{bmatrix}w_0 & w_1 & \cdots & w_M\end{bmatrix}\begin{bmatrix}\phi_0(\mathbf{x}) \\ \phi_1(\mathbf{x}) \\ \cdots \\ \phi_M(\mathbf{x})\end{bmatrix}$
		- $=\mathbf{w}^\top\mathbf{\phi}(\mathbf{x})$
	- where,
		- $M+1$ is the number of operations.
		- $\mathbf{w}$ is the weight vector:
			- $\mathbf{w}=\begin{bmatrix}w_0 \\ w_1 \\ \cdots \\ w_M\end{bmatrix}$.
		- $\mathbf{\phi}$ is the basic function vector:
			- $\mathbf{\phi}=\begin{bmatrix}\phi_1 \\ \phi_2 \\ \cdots \\ \phi_M\end{bmatrix}$.

## 8.1.2 Decide Basic Functions $\mathbf{\phi}$
### Polynomial Regression
- $\forall j\in[0,M], \ \phi_j(\mathbf{x})=\mathbf{x}^j$.

### Gaussian Basis Functions
- $\forall j\in[0,M], \ \phi_j(\mathbf{x})=e^{-\dfrac{\|\mathbf{x}-\mathbf{\mu_j}\|^2}{2\sigma^2}}$

### Spline Basis Functions
- Piecewise polynomials.