### Final Tip:
- Primal & Dual Problem
- How to re-formulate Primal to Dual Problem (easier-to-solve)
# 10.1 Hyperplane & Binary Classification
## 10.1.1 Binary Classification
**Given**
- Training Sample: $\mathcal{D}=\{(\mathbf{x}_{t}, y_{t})\}_{t=1}^{N}$
	- Instances: $\mathbf{x}_{t}\in \mathbb{R}^m$
	- Labels: $y_{t}\in\{+1,-1\}$
**Do**
- Train a prediction function: $h:\mathcal{X}\mapsto\{+1,-1\}$

Given the arbitrarily dimensioned hyperplane $\mathbf{w}^\top\mathbf{x}+b=0$, we could conclude that:
$$
\begin{cases}
\mathbf{w}^\top\mathbf{x}_{i}+b\geq {0} & \text{the point is above the plane} \\
\mathbf{w}^\top\mathbf{x}_{i}+b\leq {0} & \text{the point is below the plane} \\
\mathbf{w}^\top\mathbf{x}_{i}+b=0 & \text{the point is on the plane}
\end{cases}
$$
One intuitive solution is to give:
$$
f(\mathbf{x})=\mathbf{w}^\top \mathbf{x}+b
$$
such that:
$$
y_{t}=\begin{cases}
+1 & \text{if} \ f(\mathbf{x}_{t})>0 \\
-1 & \text{if} \ f(\mathbf{x}_{t})<0
\end{cases}
$$
From all the points with pre-defined labels, we need to find such plane $\mathbf{w}^\top\mathbf{x}+b=0$ that separates all the points into two parts correctly, completing a binary classification.
### Hyperplane Properties
The weight vector $\mathbf{w}$ is the normal vector of the hyperplane $\mathbf{w}^\top\mathbf{x}+b=0$.
**Proof:**
Suppose $\mathbf{x}_{1}$ and $\mathbf{x_{2}}$ are two points on the hyperplane. Therefore: $$f(\mathbf{x}_{1})=f(\mathbf{x}_{2})=0 $$
Namely,
$$
\begin{align}
\mathbf{w}^\top\mathbf{x}_{1}+b &= 0 \\ \\
\mathbf{w}^\top\mathbf{x}_{2}+b &= 0
\end{align}
$$
Differencing the two equations, we get:
$$
\mathbf{w}^\top(\mathbf{x}_{1}-\mathbf{x}_{2})=0
$$
$\mathbf{x}_{1}-\mathbf{x}_{2}$ is an arbitrary line in the hyperplane. 
- The above equation states that this line is perpendicular to the vector $\mathbf{w}$.

This means that $\mathbf{w}$ is perpendicular to any line in the hyperplane. 
- There doesn't exist any linear combinations of two lines from the plane that could represent the vector $\mathbf{w}$. 
- Therefore, $\mathbf{w}$ is perpendicular to the hyperplane.
### Distance Equation
We need to know the distance from an arbitrary data to the hyperplane. We express a point $\mathbf{x}$ with respect to its orthogonal projection on the hyperplane.

Consider that a point $\mathbf{x}=\mathbf{x}_{p}+\rho\frac{\mathbf{w}}{\|\mathbf{w}\|}$.
- $\mathbf{x}_{p}$ is the orthogonal projection of $\mathbf{x}$ on the hyperplane.
	- $f(\mathbf{x}_{p})=0$
- $\frac{\mathbf{w}}{\|\mathbf{w}\|}$ is the unit vector with the same direction of the normal vector $\|\mathbf{w}\|$ of the hyperplane. 
  (与平面法向量同向的单位向量)
	- $\mathbf{w}$ is the normal vector.
	- $\|\mathbf{w}\|$ is the length/norm of the vector: $\|\mathbf{w}\|=\sqrt{ \mathbf{w}^{\top}\mathbf{w} }$.
- $\rho$ is the geometric distance from the point to the hyperplane.

Therefore, 
$$
\begin{align}
f(\mathbf{x}) &= \mathbf{w}^\top\mathbf{x}+b \\
&= \mathbf{w}^\top\Bigl(\mathbf{x}_{p}+\rho\frac{\mathbf{w}}{\|\mathbf{w}\|}\Bigr)+b \\
&= \Bigl(\mathbf{w}^\top\mathbf{x}_{p}+b\Bigr)+\rho\frac{\mathbf{w}^\top\mathbf{w}}{\|\mathbf{w}\|} \\
&= 0+\rho\frac{\|\mathbf{w}\|^{2}}{\|\mathbf{w}\|} \\
&= \rho\|\mathbf{w}\|
\end{align}
$$
Namely,
$$
\rho_{\mathbf{x}}=\frac{f(\mathbf{x})}{\|\mathbf{w}\|}
$$
![[SVM.png]]
## 10.1.2 Canonical Form
Remark that the equation of the hyperplane:
$$
\mathbf{w}^\top\mathbf{x}+b=0
$$
We shift the plane up and down, getting:
$$
\begin{align}
\mathbf{w}^\top\mathbf{x}+b &= 1 \\
\mathbf{w}^\top\mathbf{x}+b &= -1
\end{align}
$$
Therefore, the set of all the points that lies on the two buffer is:
$$
|\mathbf{w}^\top\mathbf{x}+b|=1
$$
Since we want the hyperplane to be robust, we want that the all the points in $\mathcal{X}$ is out of the buffer mentioned above.
$$
\mathbf{w}^\top\mathbf{x}_{t}+b \geq 1 \lor \ \mathbf{w}^\top\mathbf{x}_{t}\leq -1 \implies |\mathbf{w}^\top\mathbf{x}+b|\geq 1
$$
This defines the canonical form of the hyperplane:
$$
\min_{\mathbf{x}_{t}\in\mathcal{X}}|\mathbf{w}^\top\mathbf{x}+{b}|=1
$$
That is, the hyperplane is fine adjusted such that:
- There always exists at least a point on its ceiling and ground buffer
- and there is no points within the buffer.

The geometric margin is given by
$$
\begin{align}
\rho &= \frac{1}{2}\Bigl(\frac{f(\mathbf{x}^{+})}{\|\mathbf{w}\|}-\frac{f(\mathbf{x}^{-})}{\|\mathbf{w}\|}\Bigr) \\
&= \frac{1}{2} \frac{1}{\|\mathbf{w}\|}\Bigl(f(\mathbf{x}^+)-f(\mathbf{x}^-)\Bigr) \\
&= \frac{1}{2} \frac{1}{\|\mathbf{w}\|}(1-(-1)) \\
&= \frac{1}{\|\mathbf{w}\|}
\end{align}
$$
![[SVM_Buffers.png]]
# 10.2 Optimization
We need a classifier that gives us the max margin.
## 10.2.1 Primal Form
We need to find an optimized weight $\mathbf{w}^*$ such that the geometric distance from one of the buffer to the hyper plane:
$$
\frac{1}{\|\mathbf{w}\|}
$$
is maximized.

Which is equivalent to *minimizing* the objective function of:
$$
\mathcal{J}(\mathbf{w})=\frac{1}{2}\|\mathbf{w}\|^2
$$
with the constraints:
$$
y_{i}\Bigl(\mathbf{w}^\top\mathbf{x}_{i}+b\Bigr)\geq 1, \ y_{i}\in\{1,-1\};\ i=1,\cdots,N
$$
Explanation:
- The function $\mathcal{J}(\mathbf{w})$ is the $l-2$ norm of the vector $\mathbf{w}$.
- The constraints regulates that all the points should be out of or on the two buffers.
## 10.2.2 Primal Lagrangian
We use the Primal Lagrangian to combine the optimization and the constraint.
The **Primal Lagrangian** is given by:
$$
\mathcal{L}(\mathbf{w}, b, \alpha)=\frac{1}{2}\|\mathbf{w}\|^{2}+\sum_{i=1}^{N}\alpha_{i}\Bigl(1-y_{i}(\mathbf{w}^\top\mathbf{x}+b)\Bigr)
$$

We introduce a Lagrange multiplier $\alpha_{i}$ for all the data samples $\mathbf{x}_{i}\in\mathcal{X}$. These multiplier terms enforce the constraint $y_{i}\Bigl(\mathbf{w}^\top\mathbf{x}+b\Bigr)\geq 1$ by:
- Penalizing the objective if the constraint is violated.
- The more it violates, the more term is added to the lagrangian function, scaled by the penalization factor of $\alpha_{i}$.
To optimize, we minimize $\mathcal{L}$ by setting $\dfrac{\partial\mathcal{L}}{\partial \mathbf{w}}=0$ and $\dfrac{\partial \mathcal{L}}{\partial b}=0$. Yielding:
$$
\begin{align}
& \mathbf{w}^{*}=\sum_{i=1}^{N}(\alpha_{i}\cdot y_{i})\cdot\mathbf{x}_{i} \\
& \sum_{i=1}^{N}\alpha_{i}\cdot y_{i} = 0
\end{align}
$$
Substituting $\mathbf{w}^{*}$ in $\mathcal{L}$ yields:
$$
\begin{align}
\mathcal{L}(\mathbf{w}^{*},b,\alpha) &= \frac{1}{2}\|\mathbf{w}^{*}\|^2+\sum_{i=1}^{N}\alpha_{i}\Bigl(1-y_{i}(\mathbf{w}^{*\top}\mathbf{x}_{i}+b)\Bigr)
\end{align}
$$
Respectively, we substitute the first term:
$$
\begin{align}
\frac{1}{2}\|\mathbf{w}^{*}\|^2 &= \frac{1}{2}\Bigl(\sum_{i=1}^{N}(\alpha_{i}\cdot y_{i})\cdot\mathbf{x}_{i}\Bigr)^\top\Bigl(\sum_{i=1}^{N}(\alpha_{i}\cdot y_{i})\cdot\mathbf{x}_{i}\Bigr) \\
&= \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}(\alpha_{i}\cdot\alpha_{j})\cdot(y_{i}\cdot y_{j})\cdot(\mathbf{x}_{i}^\top\mathbf{x}_{j})
\end{align}
$$
Then, we substitute the second term:
$$
\begin{align}
\sum_{i=1}^{N}\alpha_{i}\Bigl(1-y_{i}(\mathbf{w}^{*\top}\mathbf{x}_{i}+b)\Bigr) &= \sum_{i=1}^{N}\alpha_{i}\Bigl(1-y_{i}(\Bigl[\sum_{j=1}^{N}\alpha_{j}\cdot y_{j} \cdot \mathbf{x}_{j}\Bigr]^\top\mathbf{x}_{i}+b)\Bigr) \\
&= \sum_{i=1}^{N}\alpha_{i}-\sum_{i=1}^{N}\alpha_{i}\cdot y_{i}(\Bigl[\sum_{j=1}^{N}\alpha_{j}\cdot y_{j} \cdot \mathbf{x}_{j}\Bigr]^\top\mathbf{x}_{i}+b) \\
&= \sum_{i=1}^{N}\alpha_{i}-\sum_{i=1}^{N}\sum_{j=1}^{N}(\alpha_{i}\cdot \alpha_{j})\cdot(y_{i}\cdot y_{j})\cdot(\mathbf{x}_{i}^\top\mathbf{x}_{j})
\end{align}
$$
Therefore, the optimal function with respect to the weights $\mathbf{w}$ and the bias $b$ would be:
$$
\mathcal{L}=-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}(\alpha_{i}\cdot \alpha_{j})\cdot(y_{i}\cdot y_{j})\cdot(\mathbf{x}_{i}^\top\mathbf{x}_{j})+\sum_{i=1}^{N}\alpha_{i}
$$
The function is already optimal with respect to the $\mathbf{w}$ and $b$. Now we find the optimum penalty factors to optimize the new function with respect to the penalty factors $\alpha_{i}$:
$$
\mathcal{G}(\alpha)=-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}(\alpha_{i}\cdot \alpha_{j})\cdot(y_{i}\cdot y_{j})\cdot(\mathbf{x}_{i}^\top\mathbf{x}_{j})+\sum_{i=1}^{N}\alpha_{i}
$$
with the constraints:
$$
\sum_{i=1}^{N}\alpha_{i}y_{i}=0
$$
