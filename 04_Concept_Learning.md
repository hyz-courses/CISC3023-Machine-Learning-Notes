# 4.1 A Concept Learning Task
- Given:
	- Instances $X=\{x_1, x_2, ...\}$
	- Target Function $c: X \rightarrow {0,1}$
	- Hypothesis:
		- An example input in the format of $X$.
		- Conjunction of constraints on attributes, where each could be:
			- A specific value (e.g., $Water=Warm$);
			- Any value (e.g., $Water=Any$);
			- No value (e.g., $Water=\emptyset$).
	- Training examples $D$:
		- Positive and negative examples of target function
		- $D=\{<X_1, c(X_1)>, <X_2, c(X_2)>, ..., <X_m, c(X_m)>\}$
- Do:
	- Derive a general if-then rule to best summarize the training examples.

# 4.2 Find-S Algorithm
## 4.2.1 General-to-specific Ordering of Hypothesis
- [DEF] More-General-than-or-Equal-to
	- Let $h_j$ and $h_k$ be bool-valued functions defined over $X$.
	- Then, $h_j$ is *more_general_than_or_equal_to* $h_k$ ($h_j\ge_g h_k$) iff:
		- $\forall x\in X, h_k(x)=1 \rightarrow h_j(x)=1$
	- Moreover, $h_j$ is strictly $more_general$ than $h_k$ iff:
		- $(h_j\ge_g h_k)\land(h_k\ngeq_g h_j)$

## 4.2.2 Find-S Algorithm
- From Specific to General
Procedure:
- Initialize $h$ to the most specific hypothesis in $H$.
- for $X$ where $c(X)=1$:
	- for $x_i$ where $x_i\in X=\{x_1,x_2,...\}$:
		- if $x_i$ in $h$ doesn't satisfy $x_i$ in $X$:
			- Replace $x_i$ in $h$ with a more general constraint that's accepted by $X$.
- Output: $h$.

# 4.3 List-Then-Eliminate Algorithm
## 4.3.1 Version Space
- [DEF] A hypothesis $h$ is ***consistent*** with a set of training examples $D$ of target concept $c$ iff:
	- $Consistent(h,D)\equiv\forall\langle x,c(x)\rangle\in D, h(x)=c(x)$
	- $h$ satisfied all the input-output pairs in $D$.
- [DEF] The ***version space*** $VS_{H,D}$ with respect to:
	- Hypothesis space $H=\{h_1, h_2, ...\}$
	- Training examples $D$
- is defined as:
	- $VS_{H,D} \equiv \{h\in H|Consistent(h,D)\}$

## 4.3.2 List-Then-Eliminate Algorithm
Procedure:
- For each training example $\langle x,c(x)\rangle\in D$ do:
	- if$c(x)=0$do:
	- if$c(x)=1$ do: