# 9.1 Binary Classification

### Final Tip:
- First Question: Similar to homework
- Second Question: Basic idea, gradient calculation, maximum likelihood calculation

## 9.1.1 Logistic Function / Sigmoid Function
**Given:**
- Two classes $C=\{c_{1},c_{2}\}$
- A particular vector $\mathbf{x}\in \mathbb{R}^d$
**Do**
- Assign the vector $\mathbf{x}$ to one of the two classes.
	- Namely, to calculate the probability of $\mathbf{x}\in c_{i}$ for all $c_{i}\in C$.

From the Bayesian Rule, we derive that
$$
\begin{align}
P(c_{1}|\mathbf{x})&=\frac{P(\mathbf{x}|c_1)P(c_1)}{P(\mathbf{x})} \\ \\
&=\frac{P(\mathbf{x}|c_1)P(c_1)}{P(\mathbf{x}|c_1)P(c_1)+P(\mathbf{x}|c_2)P(c_2)} \\  \\
&=\dfrac{1}{1+\frac{P(\mathbf{x}|c_{2})P(c_{_{2}})}{P(\mathbf{x}|c_{1})P(c_{1})}} \\ \\
&=\dfrac{1}{1+e^{-\ln\Bigl[\frac{P(\mathbf{x}|c_{1})}{P(\mathbf{x}|c_{2})}\Bigr]-\ln\Bigl[\frac{P(c_{1})}{P(c_{2})}\Bigr]}}
\end{align}
$$

which can be written in the form of a **Logistic function / Sigmoid Function:**
$$P(c_{1}|\mathbf{x})=\dfrac{1}{1+e^{-\xi}}$$
where,
$$\xi=\ln\Bigl[\frac{P(\mathbf{x}|c_{1})}{P(\mathbf{x}|c_{2})}\Bigr]+\ln\Bigl[\frac{P(c_{1})}{P(c_{2})}\Bigr]$$
- Likelihood Ratio: $\ln\Bigl[\frac{P(\mathbf{x}|c_{1})}{P(\mathbf{x}|c_{2})}\Bigr]$
- Prior Ratio: $\ln\Bigl[\frac{P(c_{1})}{P(c_{2})}\Bigr]$
### Properties of Logistic Functions
```functionplot
---
title: Logistic Function
xLabel: \xi
yLabel: \sigma
bounds: [-3, 3, -0.2, 1]
disableZoom: false
grid: true
---
f(x)=1/(1+E^(-x))
```

- [i] Limits
$$
\begin{align}
\lim_{ \xi \to -\infty }\sigma(\xi) &=0  \\ \\
\lim_{ \xi \to \infty }\sigma(\xi) &=1 
\end{align}
$$
- [i] Central Symmetry
$$
\sigma(-\xi)=1-\sigma(\xi)
$$
- [i] Derivative
$$
\begin{align}
\dfrac{d}{d\xi}\sigma(\xi) &= \sigma(\xi)(-\xi) \\
&=\sigma(\xi)\Bigl(1-\sigma(\xi)\Bigr)
\end{align}
$$
*Proof.*
$$
\begin{align}
\dfrac{d}{d\xi}\sigma(\xi)&=\dfrac{d}{d\xi}\frac{1}{1+e^{-\xi}} \\ \\
&=(\dfrac{d}{d(1+e^{-\xi})}\cdot\frac{1}{1+e^{-\xi}})\cdot\frac{d(1+e^{-\xi})}{d\xi} \\ \\
&=\frac{-1}{(1+e^{-\xi})^2}\cdot(-e^{-\xi}) \\ \\
&=\frac{e^{-\xi}}{(1+e^{-\xi})^2} \\ \\
&=\frac{e^{-\xi}}{(1+e^{-\xi})}\cdot\frac{1}{(1+e^{-\xi})} \\ \\
&=\frac{1}{1+e^{\xi}}\cdot\frac{1}{1+e^{-\xi}} \\ \\
&=\sigma(-\xi)\cdot\sigma(\xi)
\end{align}
$$

## 9.1.2 Multivariate Gaussian Model for $P(\mathbf{x}|c_{i})$
Assumed that:
- within each class
- the multi-variate input vector $\mathbf{x}$ follows a Gaussian Distribution with a *common* covariate $\Sigma$.
$$P(\mathbf{x}|c_{i})=\dfrac{1}{(2\pi)^\frac{d}{2}|\Sigma|^\frac{1}{2}}e^{-\frac{1}{2}(\mathbf{x}-\mathbf{\mu}_{i})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{\mu}_{i})}$$
From which we derive that
$$
\begin{align}
\ln\Bigl(\frac{P(\mathbf{x}|c_1)}{P(\mathbf{x}|c_2)}\Bigr) &= \ln\Bigl[\frac{e^{-\frac{1}{2}(\mathbf{x}-\mu_1)^\top\Sigma^{-1}(\mathbf{x}-\mu_1)}}{e^{-\frac{1}{2}(\mathbf{x}-\mu_2)^\top\Sigma^{-1}(\mathbf{x}-\mu_2)}}\Bigr] \\ \\
&=\frac{1}{2}(\mathbf{x}-\mathbf{\mu}_{2})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{\mu}_{2})-\frac{1}{2}(\mathbf{x}-\mathbf{\mu}_{1})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{\mu}_{1}) \\ \\
&=\frac{1}{2}(\mathbf{x}^\top\Sigma^{-1}-\mathbf{\mu}_{2}^\top\Sigma^{-1})(\mathbf{x}-\mathbf{\mu}_{2})-\frac{1}{2}(\mathbf{x}^\top\Sigma^{-1}-\mathbf{\mu}_{1}^\top\Sigma^{-1})(\mathbf{x}-\mathbf{\mu}_{1}) \\ \\
&=\frac{1}{2}(\mathbf{x}^\top\Sigma^{-1}\mathbf{x}-\mathbf{x}^\top\Sigma^{-1}\mathbf{\mu}_{2}-\mathbf{\mu}_{2}^\top\Sigma^{-1}\mathbf{x}+\mathbf{\mu}_{2}^\top\Sigma^{-1}\mathbf{\mu}_{2}) \\
&-\frac{1}{2}(\mathbf{x}^\top\Sigma^{-1}\mathbf{x}-\mathbf{x}^\top\Sigma^{-1}\mathbf{\mu}_{1}-\mathbf{\mu}_{1}^\top\Sigma^{-1}\mathbf{x}+\mathbf{\mu}_{1}^\top\Sigma^{-1}\mathbf{\mu}_{1}) \\  \\
&=\frac{1}{2}\Bigl[\mathbf{x}^\top\Sigma^{-1}(\mathbf{\mu}_{1}-\mathbf{\mu}_{2})+(\mathbf{\mu}_{1}^\top-\mathbf{\mu}_{2}^\top)\Sigma^{-1}\mathbf{x}+(\mathbf{\mu}_{2}^\top+\mathbf{\mu}_{1}^\top)\Sigma^{-1}(\mathbf{\mu}_{2}-\mathbf{\mu}_{1})\Bigr] \\  \\
&=(\mathbf{\mu}_{1}-\mathbf{\mu}_{2})^\top\Sigma^{-1}\mathbf{x}+\frac{1}{2}(\mathbf{\mu}_{2}+\mathbf{\mu}_{1})^\top\Sigma^{-1}(\mathbf{\mu}_{2}-\mathbf{\mu}_{1})
\end{align}
$$

- [*] Therefore, the exponential $\xi$ can be rewritten as:
$$
\begin{align}
\xi &= \ln\Bigl(\frac{P(\mathbf{x}|c_1)}{P(\mathbf{x}|c_2)}\Bigr) + \ln\Bigl(\frac{P(c_1)}{P(c_2)}\Bigr) \\ \\
&= (\mathbf{\mu}_{1}-\mathbf{\mu}_{2})^\top\Sigma^{-1}\mathbf{x}+\frac{1}{2}(\mathbf{\mu}_{2}+\mathbf{\mu}_{1})^\top\Sigma^{-1}(\mathbf{\mu}_{2}-\mathbf{\mu}_{1})+\ln\Bigl(\frac{P(c_1)}{P(c_2)}\Bigr) \\ \\
&=\Bigl[\Sigma^{-1}(\mathbf{\mu}_{1}-\mathbf{\mu}_{2})\Bigr]^\top\mathbf{x} +\Bigl[\frac{1}{2}(\mathbf{\mu}_{2}+\mathbf{\mu}_{1})^\top\Sigma^{-1}(\mathbf{\mu}_{2}-\mathbf{\mu}_{1})+\ln\Bigl(\frac{P(c_{1})}{P(c_{2})}\Bigr)\Bigr]\\ \\
&=\mathbf{w}^\top\mathbf{x}+b
\end{align}
$$

In conclusion,
$$P(c_{1}|\mathbf{x})=\dfrac{1}{1+e^{-(\mathbf{w}^\top\mathbf{x}+b)}}$$
where
- $\mathbf{w}=\Sigma^{-1}(\mu_{1}-\mu_{2})$, and
- $b=\frac{1}{2}(\mathbf{\mu}_{2}+\mathbf{\mu}_{1})^\top\Sigma^{-1}(\mathbf{\mu}_{2}-\mathbf{\mu}_{1})+\ln\Bigl[\frac{P(c_{1})}{P(c_{2})}\Bigr]$

## 9.1.3 Maximum Likelihood Formulation
### Recall: Bernoulli Distribution
Suppose that a variable $Y$ confronts a Bernoulli Distribution,
- i.e., $Y \sim \text{Bernoulli}(p)$
- where $p$ is the probability of being $\text{Success}$,
the probabilistic distribution function is
$$
P(Y=y)=
\begin{cases}
p & \text{if} \ y = 1 \\
1-p & \text{if} \ y=0
\end{cases}
$$
### Convert $\sigma$ to Probability
We want to predict a binary output $y_{n}\in\{0,1\}$ from an input $\mathbf{x}_{n}$. From above, we know that the logistic regression has the form:
$$
y_{n}=\sigma(\mathbf{w}^\top\mathbf{x}_{n})+\epsilon_{n}
$$
where
$$\sigma(\xi)=\dfrac{1}{1+e^{-\xi}}=\frac{e^{\xi}}{1+e^\xi}$$
We model input-output by a conditional Bernoulli Distribution:
$$
P(y_{n}=y|\mathbf{x}_{n})=
\begin{cases}
\sigma(\mathbf{w}^\top\mathbf{x}_{n}) & \text{if} \ y=1 \\
1-\sigma(\mathbf{w}^\top\mathbf{x}_{n}) & \text{if} \ y=0
\end{cases}
$$
### Bernoulli Distribution Modelling
Given $\{(\mathbf{x}_{n},y_{n})|n=1,\dots,N\}$, the likelihood is given by
$$
\begin{align}
P(\mathbf{y}|\mathbf{X},\mathbf{w}) &= \prod_{n=1}^{N}P(y_n|\mathbf{x}_n) \\ \\
&=\prod_{n=1}^{N}P\Bigl(y_n=1|\mathbf{x}_n\Bigr)^{y_n}\cdot\Bigl(1-P(y_n=1|\mathbf{x}_n)\Bigr)^{1-y_n} \\ \\
&=\prod_{n=1}^{N}\sigma(\mathbf{w}^\top\mathbf{x}_{n})^{y_{n}}\Bigl(1-\sigma(\mathbf{w}^\top\mathbf{x}_{n})\Bigr)^{1-y_{n}}
\end{align}
$$
The log-likelihood function is thus given by:
$$
\begin{align}
\mathcal{L}(\mathbf{y}|\mathbf{X},\mathbf{w}) &= \log\sum_{n=1}^{N}P(y_n|\mathbf{x}_n) \\ \\
&= \sum_{n=1}^{N}\Bigl(y_n\cdot\log\Bigl[\sigma(\mathbf{w}^\top\mathbf{x}_n)\Bigr]+(1-y_n)\cdot\log\Bigl[1-\sigma(\mathbf{w}^\top\mathbf{x})\Bigr]\Bigr)
\end{align}
$$
We want to maximize this log-likelihood $\mathcal{L}$. However, $\mathcal{L}$ is not a polynomial, the calculation of the maximum of nonlinear function of $\mathbf{w}$ cannot be done in a closed form. That is, it is very costly to directly compute:
$$
\dfrac{\partial \mathcal{L}}{\partial \mathbf{w}}=0
$$
Therefore, an iterated re-weighted least squares (IRLS) is then performed, derived from the Newton's method.

# 9.2 Newton's Method
## 9.2.1 Theoretical Foundations
### Key Point 1. Gradient 梯度: Step Direction
Consider a real-valued function $f(\mathbf{x})$, which takes a real-valued vector $\mathbf{x}\in \mathbb{R}^d$ as an input:
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

### Key Point 2. Hessian Matrix 海森矩阵: Step Size
If $f(\mathbf{x})$ belongs to the class $C^2$, the Hessian matrix $\mathbf{H}$ is defined as the symmetric matrix with the combination of any two dimensions.
$$
\begin{align}
\mathbf{H}&=\nabla^2 f(\mathbf{x}) 
\\ \\
% Partial w.r.t. w over transposed 1st order gradient %
&= \dfrac{\partial}{\partial\mathbf{x}}\Bigl[\nabla f(\mathbf{x}) \Bigr]^\top
\\ \\
% Expanded First-order gradient% 
&= \dfrac{\partial}{\partial\mathbf{x}}
\begin{bmatrix}
\dfrac{\partial f(\mathbf{x})}{\partial x_1}
&
\dfrac{\partial f(\mathbf{x})}{\partial x_2}
&
\cdots
&
\dfrac{\partial f(\mathbf{x})}{\partial x_d}
\end{bmatrix}
\\ \\
% Fully-Expanded Matrix Form %
&=\begin{bmatrix}
\frac{\partial^2f}{\partial x_{1}^2} & 
\frac{\partial^2f}{\partial x_{1}\partial x_{2}} & 
\cdots & 
\frac{\partial^2f}{\partial x_{1}\partial x_{d}} 
\\
\frac{\partial^2f}{\partial x_{2}\partial x_{1}} & \frac{\partial^2f}{\partial x_{2}^2} &
\cdots & 
\frac{\partial^2f}{\partial x_{2}\partial x_{d}} 
\\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2f}{\partial x_{d}\partial x_{1}} & \frac{\partial^2f}{\partial x_{d}\partial{x_{2}}} & 
\cdots & 
\frac{\partial^2f}{\partial x_{d}^2} 
\end{bmatrix}
\end{align}
$$

>The Hessian matrix can not only help us find the extreme points of the function (through the first-order derivative is 0), but also determine whether the point is a minimum, maximum or saddle point by analyzing the curvature of the function near the extreme point. For example, if the Hessian matrix is ​​positive definite, it means that the extreme point is a local minimum.

### Key Point 3. Gradient Descent/Ascent 梯度下降、上升
The gradient descent/ascent learning is a simple first-order iterative method for minimization/maximization.

Gradient Descent: Iterative Minimization
$$
\mathbf{w}^{(k+1)}=\mathbf{w}^{(k)}-\eta\Bigl(\frac{\partial \mathcal{J}}{\partial \mathbf{w}}\Bigr)
$$
Gradient Ascent: Iterative Maximization
$$
\mathbf{w}^{(k+1)}=\mathbf{w}^{(k)}+\eta\Bigl(\frac{\partial \mathcal{J}}{\partial \mathbf{w}}\Bigr)
$$
where the learning rate $\eta>0$.

The gradient $\dfrac{\partial\mathcal{J}}{\partial \mathbf{w}}$ gives the direction of the movement of $\mathbf{w}$. The learning rate $\eta$ gives the step size.

## 9.2.2 Newton's Method
The Basic idea of Newton's method is to optimize the quadratic (二次的) approximation,
- of the objective function $\mathcal{J}(\mathbf{w})$,
- around the current point $\mathbf{w}^{(k)}$.

The Newton's Method tells us that, for an arbitrary function $\mathcal{J}: \mathbb{R}^D\mapsto\mathbb{R}$, we can iteratively approximate $\dfrac{\partial\mathcal{J}(\mathbf{w})}{\partial\mathbf{w}}=0$ by:
$$
\mathbf{w}^{\text{new}}
=
\mathbf{w}^{\text{old}}
-
\Bigl[\nabla^2\mathcal{J}(\mathbf{w}^{\text{old}})\Bigr]
\cdot
\nabla\mathcal{J}(\mathbf{w}^{\text{old}})
$$
### Key Point 1. Tayler Series of $\mathcal{J}(\mathbf{w})$
 - [i] The objective of using Taylor Series of $\mathcal{J}(\mathbf{w})$ is to:
- Locally approximate the polynomial with Tayler Series at a specific step $\mathbf{w}^{(k)}$.
The Taylor Series of a function $f(x): \mathbb{R}\mapsto\mathbb{R}$ around a point $a$ yields:
$$
\begin{align}
f(x)
&=
f(a)+\frac{f'(a)}{1!}(x-a)
+\frac{f''(a)}{2!}(x-a)^2
+\frac{f'''(a)}{3!}(x-a)^3
+\cdots
\\ \\
&=
\sum_{i=0}^{\infty}\frac{f^{(i)}(a)}{i!}(x-a)^i
\end{align}
$$
Now we consider an abstract function $\mathcal{J}(\mathbf{w})$.
$$
\mathcal{J}:\mathbb{R}^D\mapsto\mathbb{R}
$$
The second-order Taylor series expansion of $\mathcal{J}(\mathbf{w})$ at the current $\mathbf{w}^{(k)}$ gives:

$$
\mathcal{J}_{2}(\mathbf{w})=
% First Term %
\mathcal{J}(\mathbf{w}^{(k)})
+
% Second Term %
\begin{bmatrix}
\nabla \mathcal{J}(\mathbf{w}^{(k)})
\end{bmatrix}^\top(\mathbf{w}-\mathbf{w}^{(k)})
+
% Third Term %
\frac{1}{2}(\mathbf{w}-\mathbf{w}^{(k)})^\top\Bigl[\nabla^2\mathcal{J}(\mathbf{w}^{(k)})\Bigr](\mathbf{w}-\mathbf{w}^{(k)})
$$

Specifically:
- First term: $\mathcal{J}(\mathbf{w}^{(k)})$, is the value of the target function $\mathcal{J}$ with respect to the current $\mathbf{w}^{(k)}$.
- Second term: $\nabla\mathcal{J}(\mathbf{w}^{(k)})$ is the *gradient* of the target function at the current $\mathbf{w}^{(k)}$.
- Third Term: $\nabla^2\mathcal{J}(\mathbf{w}^{(k)})$ is the *Hessian Matrix* describing the target function's 2nd order derivative at current $\mathbf{w}^{(k)}$.

### Key Point 2. Differentiation of $2^{\text{nd}}$ order Taylor Series
Differentiate the above second-order Taylor Series with respect to $\mathbf{w}$, we get the approximation of the gradient function at the point $\mathbf{w}$:
$$
\nabla\mathcal{J}(\mathbf{w})=0+\nabla\mathcal{J}(\mathbf{w}^{(k)})
+\nabla^2\mathcal{J}(\mathbf{w}^{(k)})(\mathbf{w}-\mathbf{w}^{(k)})
$$
Remember that $\nabla\mathcal{J}(\mathbf{w})\in\mathbb{R}^d$. Set the gradient equal to $\mathbb{0}^d$:
$$
\begin{align}
&&\dfrac{\partial\mathcal{J}}{\partial\mathbf{w}} &= 0
\\ \\
&\implies \quad &
0 &= \nabla\mathcal{J}(\mathbf{w}^{(k)})
+
\nabla^2\mathcal{J}(\mathbf{w}^{(k)})\cdot(\mathbf{w}-\mathbf{w}^{(k)}) 
\\ \\
&\implies \quad &
0 &= 
\nabla\mathcal{J}(\mathbf{w}^{(k)})
+
\nabla^2\mathcal{J}(\mathbf{w}^{(k)})\cdot\mathbf{w}-\nabla^2\mathcal{J}(\mathbf{w}^{(k)})\cdot\mathbf{w}^{(k)} 
\\ \\
&\implies \quad &
\nabla^2\mathcal{J}(\mathbf{w}^{(k)})\mathbf{w} &= \nabla^2\mathcal{J}(\mathbf{w}^{(k)})\cdot\mathbf{w}^{(k)}
-
\nabla\mathcal{J}(\mathbf{w}) 
\\ \\
&\implies \quad &
\mathbf{w}^{(k+1)}
&=
\mathbf{w}^{(k)}
-
\Bigl[\nabla^2\mathcal{J}(\mathbf{w}^{(k)})\Bigr]^{-1}
\cdot
\nabla\mathcal{J}(\mathbf{w}^{(k)})
\end{align}
$$

> Note that the learning rate here is the inverse Hessian matrix.

At this point, we conclude that, for an arbitrary function $\mathcal{J}$ that maps a $d$-dimensional vector to a scalar. 

We can compute the *optimal vector* that produces the *optimum scalar* by:
1. Calculate its gradient at the current step.
2. Calculate its Hessian at current step.
3. Apply Newton's method to get the vector of the next step.

# 9.3 Logistic Regression Algorithms
- [I] Remark: We have already computed the log-likelihood $\mathcal{L}(\mathbf{y}|\mathbf{X},\mathbf{w})$ under the given data set $\mathbf{X},\mathbf{y}$.

Here, $\mathcal{L}$ is an example of the abstract function $\mathcal{J}$ since:
$$
\mathcal{L}: \mathbb{R}^d (\text{weight } \mathbf{w})\mapsto\mathbb{R} (\text{Probability})
$$
- We need to find an optimal vector that produces the maximum scalar.
- But calculating $\dfrac{\partial\mathcal{L}}{\partial\mathcal{w}}$ is very costly.
- Therefore, we could have a work-around using Newton's method.

The gradient Ascent Learning has the form
$$
\mathbf{w}^{\text{new}}=\mathbf{w}^{\text{old}}+\eta\Bigl(\dfrac{\partial\mathcal{L}}{\partial \mathbf{w}}\Bigr)
$$
## 9.3.1 Calculate Gradient
Recall the log-likelihood:
$$
\mathcal{L}=
\sum_{n=1}^{N}\Bigl[y_{n}\log \sigma(\mathbf{w}^\top\mathbf{x}_{n})
+
(1-y_{n})\log\Bigl(1-\sigma(\mathbf{w}^\top\mathbf{x}_{n})\Bigr)\Bigr]
$$
Take its first order derivative as the gradient:
$$
\dfrac{\partial\mathcal{L}}{\partial \mathbf{w}}=\sum_{n=1}^{N}\Bigl[y_{n}\frac{\sigma_{n}'}{\sigma_{n}}\mathbf{x}_{n}+(1-y_{n})\frac{-\sigma_{n}'}{(1-\sigma_{n})}\mathbf{x}_{n}\Bigr]
$$
Note that using chain rule:
$$
\begin{align}
\dfrac{\partial}{\partial \mathbf{w}} \log\sigma(\mathbf{w}^\top\mathbf{x}_{n})
&=
\dfrac{\partial \mathcal{L}}{\partial\sigma}
\cdot
\dfrac{\partial\sigma}{\partial(\mathbf{w}^\top\mathbf{x}_n)}
\cdot
\dfrac{\partial\mathbf{w}^\top\mathbf{x}_n}{\partial\mathbf{w}}
\\ \\
&= 
\dfrac{1}{\sigma(\mathbf{w}^\top\mathbf{x}_{n})}
\cdot
\sigma'(\mathbf{w}^\top\mathbf{x}_n)
\cdot
\mathbf{x}_n
\end{align}
$$

By using the 2nd and 3rd property of the logistic function $\sigma$, we obtain:
$$
\begin{align}
\dfrac{\partial\mathcal{L}}{\partial \mathbf{w}}
&=
\sum_{n=1}^{N}\Bigl[y_{n}\frac{\sigma_{n}(1-\sigma_{n})}{\sigma_{n}}\mathbf{x}_{n}+(1-y_{n})\frac{-\sigma_{n}(1-\sigma_{n})}{1-\sigma_{n}}\mathbf{x}_{n}\Bigr] 
\\ \\
&=\sum_{n=1}^{N}\Bigl(y_{n}(1-\sigma_{n})\mathbf{x}_{n}-(1-y_{n})\sigma_{n}\mathbf{x}_{n}\Bigr)
\\ \\
&=\sum_{n=1}^{N}\Bigl[y_{n}(1-\sigma_{n})-(1-y_{n})\sigma_{n}\Bigr]\mathbf{x}_{n}
\\ \\
&= \sum_{n=1}^{N}\Bigl(y_{n}-y_{n}\sigma_{n}-\sigma_{n}+y_{n}\sigma_{n}\Bigr)\mathbf{x}_{n} 
\\ \\
&= \sum_{n=1}^{N}(y_{n}-\sigma_{n})\mathbf{x}_{n} \in\mathbb{R}^D
\end{align}
$$

Lastly, it could be concluded that:
$$
\dfrac{\partial\mathcal{L}}{\partial \mathbf{w}}
=
\sum_{n=1}^{N}\Bigl(y_{n}-\sigma(\mathbf{w}^\top\mathbf{x}_{n})\Bigr)\mathbf{x}_{n}
$$
As discussed before, it is a vector with the same shape of $\mathbf{x}_{n}$.

- [*] By now we know that we could update $\mathbf{w}$ by:
$$
\mathbf{w}^{(k+1)}
=\mathbf{w}^{(k)}+\eta \cdot \sum_{n=1}^{N}\Bigl[y_{n}-\sigma\Bigl(\mathbf{{w^{(k)}}^\top\mathbf{x}_{n}}\Bigr)\Bigr]\mathbf{x}_{n}
$$

> 记到这里就行了，学习率老师会给所以海森矩阵不用手算。

### Example: Gradient Calculation
**Given:**
- Historical data:
	- $x_{1}=\begin{pmatrix}2\\1\end{pmatrix}, x_{2}=\begin{pmatrix}1\\2\end{pmatrix},x_{3}=\begin{pmatrix}3\\3\end{pmatrix}$
	- $y_{1}=1,y_{2}=0,y_{3}=0$
- Probability:
	- $P(\mathbf{x}|\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x})$.
- Initial weight:
	- $\mathbf{w}_{0}=\begin{bmatrix}0 \\ 0\end{bmatrix}$.
- Learning rate:
	- $\eta=0.1$
**Do:**
- Update weight for one step.
$$
\begin{align}
\mathbf{w}_{k+1}&=\mathbf{w}_{k}+\eta \cdot \sum_{n}(y_{n}-\sigma(\mathbf{w}^\top\mathbf{x}_{n}))\mathbf{x}_{n} \\ \\
\mathbf{w}_{1} &=\begin{bmatrix}0\\0\end{bmatrix}+0.1\cdot \Bigl[
(1-0.5)\begin{pmatrix}2\\1\end{pmatrix}+(0-0.5)\begin{pmatrix}1\\2\end{pmatrix}+(0-0.5)\begin{pmatrix}3\\3\end{pmatrix}
\Bigr]\\ \\
&= \begin{bmatrix}0\\0\end{bmatrix}+0.1\cdot\begin{bmatrix}1-0.5-1.5\\0.5-1-1.5\end{bmatrix}\\ \\
&= \begin{bmatrix}0\\0\end{bmatrix}+0.1\cdot\begin{bmatrix}-1\\-2\end{bmatrix}\\ \\
&= \begin{bmatrix}-0.1\\-0.2\end{bmatrix}
\end{align}
$$

## 9.3.2 Calculate Hessian
Calculate the Hessian:
$$
\mathbf{H}=\nabla^2\mathcal{L}
$$
Differentiate every term in the gradient:
$$
\begin{align}
\mathbf{H}
&= \nabla^2\mathcal{L}
\\ \\
&= \dfrac{\partial}{\partial\mathbf{w}}\Bigl[\sum_{n=1}^{N}(y_n-\sigma_n)\mathbf{x}_n \Bigr]^\top
\\ \\
&= \dfrac{\partial}{\partial\mathbf{w}}\sum_{n=1}^{N}(y_n-\sigma_n)\mathbf{x}_n^\top
\\ \\
&= \sum_{n=1}^{N}\dfrac{\partial}{\partial\mathbf{w}}(y_n-\sigma_n)\mathbf{x}_n^\top
\\ \\
&= \sum_{n=1}^{N}-\sigma_n'\mathbf{x}_n^\top
\\ \\
&= \sum_{n=1}^{N}-\sigma_n(1-\sigma_n)\mathbf{x}_n\mathbf{x}_n^\top
\end{align}
$$
In general:
$$
\nabla^2\mathcal{L}=\sum_{n=1}^{N}-\sigma_{n}(1-\sigma_{n})\mathbf{x}_{n}\mathbf{x}_{n}^\top
$$
## 9.3.3 Objective Function
Notice that the original Log-Likelihood:
$$\mathcal{L}(\mathbf{y}|\mathbf{X},\mathbf{w})=\log\sum_{n=1}^{n} P(y_{n}|\mathbf{x}_{n})$$
is *negative* since the probability $P(y_{n}|\mathbf{x}_{n})$ is lower than $1$. 

Therefore, we set the objective function $\mathcal{J}(\mathbf{w})$ to be the negative log-likelihood:
$$
\mathcal{J}(\mathbf{w})=-\mathcal{L}(\mathbf{w})=-\sum_{n=1}^{N}\Bigl[y_{n}\log \Bigl(\sigma(\mathbf{w}^\top\mathbf{x}_{n})\Bigr)+(1-y_{n})\log\Bigl(1-\sigma(\mathbf{w}^\top\mathbf{x}_{n})\Bigr)\Bigr]
$$
Therefore,
- The gradient: $\nabla\mathcal{J}(\mathbf{w})=-\sum_{n=1}^{N}(y_{n}-\sigma_{n})\mathbf{x}_{n}$
- The Hessian: $\nabla^2\mathcal{J}(\mathbf{w})=\sum_{n=1}^{N}\sigma_{n}(1-\sigma_{n})\mathbf{x}_{n}\mathbf{x_{n}}^\top$

The optimization problem went from:
- Maximizing $\mathcal{L}$
- to minimizing $-\mathcal{L}$

Therefore, we use gradient descent:
$$
\mathbf{w}^{\text{new}} = \mathbf{w}^{\text{old}}-\eta\cdot\dfrac{\partial (-\mathcal{L})}{\partial\mathbf{w}}
$$

The *update* part of the Newton's method $\eta\Bigl(\dfrac{\partial\mathcal{L}}{\partial \mathbf{w}}\Bigr)$ has the form:
$$
\Delta \mathbf{w}= - \Bigl[\sum_{n}^{N}\sigma_{n}(1-\sigma_{n})\mathbf{x}_{n}\mathbf{x}_{n}^\top\Bigr]^{-1}\Bigl[-\sum_{n=1}^{N}(y_{n}-\sigma_{n})\mathbf{x}_{n}\Bigr]
$$
Namely,
$$
\Delta \mathbf{w} 
= 
\Bigl(
\mathbf{X} \ 
S \ 
\mathbf{X}^\top\Bigr)^{-1}
Sb
$$
where:
$$
\begin{align}
\\ \\
\mathbf{X}
&=
\begin{bmatrix}
- & \mathbf{x}_1^\top & - \\
- & \mathbf{x}_2^\top & - \\
& \vdots & \\
- & \mathbf{x}_d^\top & -
\end{bmatrix}
\\ \\
S 
&=
\begin{bmatrix}
\sigma_{1}(1-\sigma_{1}) & 0 & \cdots & 0 \\ 
0 & \sigma_{2}(1-\sigma_{2}) & \cdots & 0 \\
\vdots & \vdots & \ddots & 0 \\ 
0 & 0 & \cdots & \sigma_{n}(1-\sigma_{n})
\end{bmatrix}
\\ \\
b
&=
\begin{bmatrix}\frac{y_{1}-\sigma_{1}}{\sigma_{1}(1-\sigma_{1})}\\\frac{y_{2}-\sigma_{2}}{\sigma_{2}(1-\sigma_{2})}\\\vdots\\\frac{y_{N}-\sigma_{N}}{\sigma_{N}(1-\sigma_{N})}\end{bmatrix}
\end{align}
$$

## 9.3.4 Recap: IRLS Algorithm
**Input**
- $\{(\mathbf{x}_{n}, y_{n})|n=1,2,\cdots,N\}$
**Do**
1. Initialize $\mathbf{w}=0$ and $w_{0}=\log\frac{\bar{\mathbf{y}}}{1-\bar{\mathbf{y}}}$
2. Repeat until convergence:
	1. for $n=1,2,\cdots,N$ do:
		1. Compute $\sigma_{n}=\sigma(\mathbf{w}^\top\mathbf{x}_{n}+w_{0})$
		2. Compute $s_{n}=\sigma_{n}(1-\sigma_{n})$
		3. Compute $b_{n}=\frac{y_{n}-\sigma_{n}}{s_{n}}$
	2. Construct $S=diag(s_{1:N})$
	3. Update $\mathbf{w}=(XSX^\top)Sb$
**Output**
- $\mathbf{w}$

# 9.4 Multi-Class Extension
## 9.4.1 Model IO
In the multi-class classification essence, we will give each class its own weight. Suppose that we have $M$ classes, the weights are:
$$
\{\mathbf{w}_{1}, \mathbf{w}_{2},\dots,\mathbf{w}_{M}\}
$$
Therefore, given an input $\mathbf{x}_{n}$, it will generate a score for each class:
$$
\text{scores}=\{\mathbf{w}_{1}^\top\mathbf{x}_{n},\mathbf{w}_{2}^\top\mathbf{x}_{n},\dots,\mathbf{w}_{M}^\top\mathbf{x}_{n}\}\subset \mathbb{R}
$$
These individual scores $\theta_{k}=\mathbf{w}_{k}^\top\mathbf{x}_{n}$ are called "Logits". We softmax these logits to make it sum to $1$:
$$
p(y_{n}=k|\mathbf{x}_{n})=\text{softmax}(\mathbf{w}_{k}^\top\mathbf{x}_{n})=\frac{e^{\mathbf{w}^\top\mathbf{x}_{n}}}{\sum_{j=1}^{M}e^{\mathbf{w}_{j}^\top\mathbf{x}_{n}}}
$$
## 9.4.2 Likelihood Function
....

# CISC3023 Assignment 3
## 1. Question 1
Given historical data as below. If current weights $\mathbf{w}$ for the logistic regression model (in which $P(y=1|\mathbf{{x}})=\sigma(\mathbf{{w}}^\top\mathbf{{x}})$) is $\begin{bmatrix}0\\0\end{bmatrix}$. Update $\mathbf{w}$ for one step according to the gradient ascent. The learning rate is $\eta=0.1$.

| $y$ | $x_{1}$ | $x_2$ | index |
| --- | ------- | ----- | ----- |
| 1   | 2       | 1     | 1     |
| 0   | 1       | 2     | 2     |
| 0   | 3       | 3     | 3     |

*Answer*:
Use Gradient Ascend:
$$
\mathbf{w}^{(k+1)}
=
\mathbf{w}^{(k)}
+
\eta \cdot \sum_{n=1}^{3}\Bigl[y_{n}-\sigma\Bigl((\mathbf{w}^{(k)})^\top\mathbf{x}_{n}\Bigr)\Bigr]\mathbf{x}_{n}
$$
Summation term:
$$
\mathbf{w}_{1}=0.1\times\Bigl[\Bigl(1-\sigma(0)\Bigr)\begin{bmatrix}2\\1\end{bmatrix}+\Bigl(0-\sigma(0)\Bigr)\begin{bmatrix}1\\2\end{bmatrix}+\Bigl(0-\sigma(0)\Bigr)\begin{bmatrix}3\\3\end{bmatrix}\Bigr]
$$
$$
=0.1\times\Bigl[\frac{1}{2}\begin{bmatrix}2\\1\end{bmatrix}-\frac{1}{2}\begin{bmatrix}1\\2\end{bmatrix}-\frac{1}{2}\begin{bmatrix}3\\3\end{bmatrix}\Bigr]
$$
$$
=0.1\times\begin{bmatrix}
1-\frac{1}{2}-\frac{3}{2} \\
\frac{1}{2}-1-\frac{3}{2}
\end{bmatrix}
$$

$$
=\begin{bmatrix}
-0.1 \\
-0.2
\end{bmatrix}
$$
That is, the updated one step of $\mathbf{w}$ is:
$$
\mathbf{w}_{1}=\begin{bmatrix}
-0.1 \\
-0.2
\end{bmatrix}
$$
