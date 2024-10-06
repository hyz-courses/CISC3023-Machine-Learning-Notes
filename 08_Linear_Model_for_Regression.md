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
- A labeling relation:
	- $D=\{(\mathbf{x}_1,y_1),(\mathbf{x}_2,y_2),\cdots,(\mathbf{x}_N,y_N)\}$
**Goal**
- To learn a mapping
	- $f(x): \mathbf{X}\rightarrow\mathbf{y}$
	- which associates $\mathbf{X}$ with $\mathbf{y}$,
- Such that we could make prediction
	- about $y_*$
	- on an unseen input $\mathbf{x}_*$ is encountered.

### To sum up
- From given relations, learn a function that
	- Takes a vector as an input
	- Output a real number that
		- "Fits" the given pattern
## 8.0.1 Definition

- [?] What is Regression?
- Regression aims at modeling the dependence of:
	- a response $Y$,
	- on a covariate $X$.
- That is, to predict the value of one or more continuous target variables $y$ given the value of input vector $x$.

- [i] The regression model is described by
$$
y=f(\mathbf{x})+\epsilon
$$



# 8.1 Linear Regression