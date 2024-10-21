# 7.0 Instance-Based Learning & KNN
## 7.0.0 Instance-Based Learning
- [i] Instance-Based Learning
- Simply *stores* training examples, i.e. features-label pairs
	- In contrast to constructing a general, explicit description of the target function over the training examples.
- *Delayed/Lazy Learning*
	- Only generalize when an unseen data needs to be classified.
	- Each time an unseen data sample is encountered, its *relationship* to the previously stored examples is examined to assign a target function value.
		- The relationships can be estimated *locally*.

## 7.0.1 K-Nearest Neighbor
- Most basic instance-based method
- Inputs of data are numeric ones:
	- Each data point is of $n$-dimensions, lying in space $\mathbb{R}^n$.
	- Define "nearest" neighbors in terms of **Euclidean Distance**.

# 7.1 Euclidean Distance
**Given:**
- Two arbitrary instances $x_i,x_j\in X=\{x_1,x_2,\cdots,x_k\}$.
	- Where $x_i,x_j$ are $n$-d datas
	- $x_i=\begin{bmatrix}x_{i1} & x_{i2} & \cdots & x_{in} \end{bmatrix}$, $x_j=\begin{bmatrix}x_{j1} & x_{j2} & \cdots & x_{jn} \end{bmatrix}$
**Do:**
- The euclidean distance between $x_i$ and $x_j$ is:
	- $d(x_i,x_j)=\sqrt{\sum_{r=1}^{n}(x_{ir}-x_{jr})^2}$

# 7.2 Output Type
## 7.2.1 Discrete Valued - Classification
**Objective:**
- Learn a discrete-valued target functions
	- of form $\mathbb{R}^n\rightarrow Y$, where
	- $Y=\{y_1,y_2,\cdots,y_s\}$ is the set of target classes

**Given:**
- Training Data-Label Pairs:
	- $D=\{\langle\mathbf{x}_{1},f(\mathbf{x}_{1})\rangle,\langle\mathbf{x}_{2},f(\mathbf{x}_{2})\rangle,\cdots,\langle\mathbf{x}_{m},f(\mathbf{x}_{m})\rangle\}\subset X\times Y$, where
		- $X$ is the set of training values:
			- $X=\{x_1, x_2, \cdots, x_m\}\subset \mathbb{R}^n$, where 
				- $\forall i\in [1,m], x_i\in\mathbb{R}^n$.
		- A set of classes:
			- $Y=\{y_1,y_2,\cdots,y_s\}$.
		- A mapping or assignments function from any training sample to a class:
			- $f:X\mapsto Y$.
- A sample query instance $x_q$ to be classified.
	- $x_q=\begin{bmatrix}x_{q1} & x_{q2} & \cdots & x_{qn}\end{bmatrix} \in \mathbb{R}^n$.
- A constant $k$.
**Do:**
- Let $\{x_1,x_2,\cdots,x_k\}$ be $k$ instances from training examples that's nearest to $x_q$.
- Output:
	- $\hat{f}(x_1)\leftarrow \text{argmax}_{y\in Y} \sum_{i=1}^{k}\delta(y,f(x_i))$, where
		- $\delta(y,f(x_i)) = \biggl\{_{0, \ if \ f(x_i)\neq y}^{1, \ if \ f(x_i)=y}$
	- Gives the *most common* value (class) from the $k$ samples.

## 7.2.2 Real-Valued - Regression
**Objective:**
- Learn a discrete-valued target functions
	- of form $\mathbb{R}^n\rightarrow y$, where
	- $y\in\mathbb{R}$, which is a real value, i.e., a scalar. 

**Given:**
- Training Data-Label Pairs:
	- $D=\{\langle\mathbf{x}_{1},y_{1}\rangle,\langle\mathbf{x}_{2},y_{1}\rangle,\cdots,\langle\mathbf{x}_{m},y_{1}\rangle\}\subset X\times \mathbb{R}$, where
		- $X$ is the set of training values:
			- $X=\{x_1, x_2, \cdots, x_m\}\subset \mathbb{R}^n$, where 
				- $\forall i\in [1,m], x_i\in\mathbb{R}^n$.
- A sample query instance $x_q$ to be classified.
	- $\mathbf{x}_q=\begin{bmatrix}x_{q1} & x_{q2} & \cdots & x_{qn}\end{bmatrix} \in \mathbb{R}^n$.
- A constant $k$.

**Do:**
- Let $\{x_1,x_2,\cdots,x_k\}$ be $k$ instances from training examples that's nearest to $x_q$.
- Output:
	- $\hat{f}(x_q)\leftarrow\dfrac{\sum_{i=1}^{k}f(x_i)}{k}$
	- That is, the *simple mean* of the values around.

# 7.3 Distance Weighted
- Weight the contribution
	- of each of the $k$ neighbors
	- according to the distance to query point $x_q$
	- closer neighbors = greater weights

## 7.3.1 Discrete-Valued
$$\hat{f}(\mathbf{x}_{q})\leftarrow \text{argmax}_{y\in Y}\sum_{i=1}^{k}w_{i}\delta(y,f(x_{i}))$$
- $=\text{argmax}_{y\in Y}\sum_{i=1}^{k}\dfrac{\delta(y,f(\mathbf{x}_{i}))}{d(\mathbf{x}_{q},\mathbf{x}_{i})^2}$
  
- $=\text{argmax}\sum_{i=1}^{k}\sum_{i=1}^{k}\dfrac{\delta(y,f(\mathbf{x}_{i}))}{\sum_{j=1}^{n}(x_{ij}-x_{qj})^2}$
where,
- $w_i=\dfrac{1}{d(x_q,x_i)^2}=\dfrac{1}{\sum_{j=1}^{n}(x_{ij}-x_{qj})^2}$

## 7.3.2 Real-Valued
The weighted mean:
$$
\hat{h}(\mathbf{x}_{q})\leftarrow \dfrac{\sum_{i=1}^{k}w_{i}f(\mathbf{x}_{i})}{\sum_{i=1}^{k}w_{i}}
$$
- $w_i=\dfrac{1}{d(x_q,x_i)^2}=\dfrac{1}{\sum_{j=1}^{n}(x_{ij}-x_{qj})^2}$

### Example
**Given**
- A set of values:

| ID               | $x_{1}$ | $x_{2}$ | $x_{3}$ | label |
| ---------------- | ------- | ------- | ------- | ----- |
| $\mathbf{x}_{1}$ | 6       | 4       | 2       | 1     |
| $\mathbf{x}_{2}$ | 2       | 8       | 3       | 9     |
| $\mathbf{x}_{3}$ | 9       | 2       | 1       | 5     |
| $\mathbf{x}_{4}$ | 3       | 8       | 6       | 1     |
| $\mathbf{x}_{5}$ | 4       | 2       | 9       | 8     |
- A query $\mathbf{x}_{q}=(3,7,3)$

*Q1.* Calculate $\hat{f}(\mathbf{x}_{q})$ using distance-weighted $k$-Nearest Neighbor for discrete valued target function.

$d^2(\mathbf{x}_{1},\mathbf{x}_{q})=(6-3)^2+(4-7)^2+(2-3)^2=19,  \ w_{1}=\dfrac{1}{19}$

$d^2(\mathbf{x}_{2},\mathbf{x}_{q})=(2-3)^2+(8-7)^2+(3-3)^2=2,  \ w_{2}=\dfrac{1}{2}$

$d^2(\mathbf{x}_{3},\mathbf{x}_{q})=(9-3)^2+(2-7)^2+(1-3)^2=65,   \ w_{3}=\dfrac{1}{65}$

$d^2(\mathbf{x}_{4},\mathbf{x}_{q})=(3-3)^2+(8-7)^2+(6-3)^2=10,   \ w_{4}=\dfrac{1}{10}$

$d^2(\mathbf{x}_{5},\mathbf{x}_{q})=(4-3)^2+(2-7)^2+(9-3)^2=62,   \ w_{5}=\dfrac{1}{62}$

Therefore, the $3$ nearest data are: 
$$
\mathbf{x}_{2},\mathbf{x}_{4},\mathbf{x}_{1}
$$
From nearest to furthest.

$vote(1)=\dfrac{1}{10}+\dfrac{1}{19}=0.1526$
$vote(9)=\dfrac{1}{2}=0.5$

Therefore, the vote is $9$.

*Q2.* Calculate $\hat{f}(\mathbf{x}_{q})$ using distance-weighted $k$-Nearest Neighbor for real valued target function.

$\hat{f}(\mathbf{x}_{q})=\dfrac{\dfrac{1}{2}\times 9+\dfrac{1}{10}\times 1+\dfrac{1}{19}\times 1}{\dfrac{1}{2}+\dfrac{1}{10}+\dfrac{1}{19}}=7.1290$
