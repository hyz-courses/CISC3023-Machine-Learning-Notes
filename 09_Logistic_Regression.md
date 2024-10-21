# 9.1 Binary Classification
## 9.1.1 Problem Setup
**Given**
- Two classes $C=\{c_{1},c_{2}\}$
- A particular vector $\mathbf{x}\in \mathbb{R}^d$
**Do**
- Assign the vector $\mathbf{x}$ to one of the two classes.

From the Bayesian Rule, we derive that
- $P(c_{1}|\mathbf{x})=\dfrac{P(\mathbf{x}|c_{1})P(c_{1})}{P(\mathbf{x})}$
	- $= \dfrac{P(\mathbf{x}|c_{1})P(c_{1})}{P(\mathbf{x}|c_{1})P(c_{1})+P(\mathbf{x}|c_{2})P(c_{2})}$
	- $=\dfrac{1}{1+\frac{P(\mathbf{x}|c_{2})P(c_{_{2}})}{P(\mathbf{x}|c_{1})P(c_{1})}}$
	- $=\dfrac{1}{1+e^{-\ln\Bigl[\frac{P(\mathbf{x}|c_{1})}{P(\mathbf{x}|c_{2})}\Bigr]-\ln\Bigl[\frac{P(c_{1})}{P(c_{2})}\Bigr]}}$
which can be written in the form of a **logistic function**:
$$P(c_{1}|\mathbf{x})=\dfrac{1}{1+e^{-\xi}}$$
where
$$\xi=\ln\Bigl[\frac{P(\mathbf{x}|c_{1})}{P(\mathbf{x}|c_{2})}\Bigr]+\ln\Bigl[\frac{P(c_{1})}{P(c_{2})}\Bigr]$$
## 9.1.2 Class-Conditional Density Function
Assume that the class-conditional densities of the multi-variate input vector $\mathbf{x}$ is a Gaussian distribution with a common covariate.
$$P(\mathbf{x}|c_{i})=\dfrac{1}{(2\pi)^\frac{d}{2}|\Sigma|^\frac{1}{2}}e^{-\dfrac{1}{2}(\mathbf{x}-\mathbf{\mu}_{i})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{\mu}_{i})}$$
From which we derive that
$$P(c_{1}|\mathbf{x})=\dfrac{1}{1+e^{-(\mathbf{w}^\top\mathbf{x}+b)}}$$
where
- $\mathbf{w}=\Sigma^{-1}(\mu_{1}-\mu_{2})$, and
- $b=\dfrac{1}{2}(\mathbf{\mu}_{2}+\mathbf{\mu}_{1})^\top\Sigma^{-1}(\mathbf{\mu}_{2}-\mathbf{\mu}_{1})+\ln\Bigl[\dfrac{P(c_{1})}{P(c_{2})}\Bigr]$

## 9.1.3 Maximum Likelihood Formulation
### Convert $\sigma$ to Probability
We want to predict a binary output $y_{n}\in\{0,1\}$ from an input $\mathbf{x}_{n}$. From above, we know that the logistic regression has the form:
$$
y_{n}=\sigma(\mathbf{w}^\top\mathbf{x}_{n})+\epsilon_{n}
$$
where
$$\sigma(\xi)=\dfrac{1}{1+e^{-\xi}}$$
We model input-output by a conditional Bernoulli Distribution:
$$P(y_{n}=1|\mathbf{x}_{n})=\sigma(\mathbf{w}^\top\mathbf{x}_{n})$$
This is the probability $k$ in the Bernoulli Distribution.
### Bernoulli Distribution Modelling
Given $\{(\mathbf{x}_{n},y_{n})|n=1,\dots,N\}$, the likelihood is given by
$$P(\mathbf{y}|\mathbf{X},\mathbf{w})=\prod_{n=1}^{N}P(y_{n}|\mathbf{x}_{n})$$
$$
=\prod_{n=1}^{N}p(y_{n}=1|\mathbf{x}_{n})^{y_{n}}(1-P(y_{n}=1|\mathbf{x}_{n}))^{1-y_{n}}
$$
$$=\prod_{n=1}^{N}\sigma(\mathbf{w}^\top\mathbf{x}_{n})^{y_{n}}(1-\sigma(\mathbf{w}^\top\mathbf{x}_{n}))^{1-y_{n}}$$

The log-likelihood function is thus given by:
$$
\mathcal{L}(\mathbf{y}|\mathbf{X},\mathbf{w})=\sum_{n=1}^{n}\log P(y_{n}|\mathbf{x}_{n})
$$
$$
=\sum_{n=1}^{N}\Bigl[y_{n}\log\sigma_{n}+(1-y_{n})\log(1-\sigma_{n})\Bigr]
$$
where $\sigma_{n}=\sigma(\mathbf{w}^\top\mathbf{x}_{n})$.

We want to maximize this log-likelihood $\mathcal{L}$. However, the calculation of the maximum of nonlinear function of $\mathbf{w}$ cannot be done in a closed form. That is, it is very costly to directly compute:
$$
\dfrac{\partial \mathcal{L}}{\partial \mathbf{w}}=0
$$
Therefore, an iterated re-weighted least squares (IRLS) is then performed, derived from the Newton's method.
# 9.2 Math Basics
## 9.2.1 Gradient
Consider a real-valued function $f(x)$, which takes a real-valued vector $\mathbf{x}\in \mathbb{R}^d$ as an input:
$$
f(\mathbf{x}):\mathbb{R}^d\mapsto\mathbb{R}
$$
The gradient of $f(\mathbf{x})$ is defined by:
$$\nabla f(\mathbf{x})=\dfrac{\partial f(\mathbf{x})}{\partial \mathbf{x}}=\begin{bmatrix}
\frac{\partial f}{\partial x_{1}} \\
\frac{\partial f}{\partial x_{2}} \\
\vdots \\
\frac{\partial f}{\partial x_{d}}
\end{bmatrix}$$
Which is the partial derivative of $f(\mathbf{x})$ with respect to all the dimensions of $\mathbf{x}$.
## 9.2.2 Hessian Matrix
If $f(\mathbf{x})$ belongs to the class $C^2$, the Hessian matrix $\mathbf{H}$ is defined as the symmetric matrix with the combination of any two dimensions.
$$\mathbf{H}=\nabla^2 f(\mathbf{x})$$
$$=\begin{bmatrix}
\dfrac{\partial^2f(\mathbf{x})}{\partial x_{i}\partial x_{j}}
\end{bmatrix}$$
$$
=\begin{bmatrix}
\frac{\partial^2f}{\partial x_{1}^2} & \frac{\partial^2f}{\partial x_{1}\partial x_{2}} & \cdots & \frac{\partial^2f}{\partial x_{1}\partial x_{d}} \\
\frac{\partial^2f}{\partial x_{2}\partial x_{1}} & \frac{\partial^2f}{\partial x_{2}^2} & \cdots & \frac{\partial^2f}{\partial x_{2}\partial x_{d}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2f}{\partial x_{d}\partial x_{1}} & \frac{\partial^2f}{\partial x_{d}\partial{x_{2}}} & \cdots & \frac{\partial^2f}{\partial x_{d}^2} 
\end{bmatrix}
$$
$$
=\dfrac{\partial}{\partial \mathbf{x}}\begin{bmatrix}
\dfrac{\partial f}{\partial \mathbf{x}}
\end{bmatrix}^\top
$$
$$
=\dfrac{\partial}{\partial \mathbf{x}}\begin{bmatrix}
\nabla f(\mathbf{x})
\end{bmatrix}^\top
$$
The Hessian matrix can not only help us find the extreme points of the function (through the first-order derivative is 0), but also determine whether the point is a minimum, maximum or saddle point by analyzing the curvature of the function near the extreme point. For example, if the Hessian matrix is ​​positive definite, it means that the extreme point is a local minimum.

## 9.2.3 Gradient Descent/Ascent 梯度下降、上升


