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

**Example:**

| Training Samples                                                     | S                                                                               | Description                                                                                           |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
|                                                                      | $S_0=\langle\emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset\rangle$ |                                                                                                       |
| $x_1=\langle Sunny, Warm, Normal,$<br>$Strong, Warm, Same\rangle, +$ | $S_1=\langle Sunny, Warm, Normal,$<br>$Strong, Warm, Same\rangle$               |                                                                                                       |
| $x_2=\langle Sunny, Warm, High,$<br>$Strong, Warm, Same\rangle, +$   | $S_2=\langle Sunny, Warm, ?,$<br>$Strong, Warm, Same\rangle$                    | $Normal\neq High$, $h$ is not consistent with $x_2$ at position $Humid$.                              |
| $x_3=\langle Rainy, Cold, High,$<br>$Strong, Warm, Change\rangle, -$ | $S_3=S_2$                                                                       | Ignore negative training samples.                                                                     |
| $x_4=\langle Sunny, Warm, High,$<br>$Strong, Cool, Change\rangle, +$ | $S_4=\langle Sunny, Warm, ?,$<br>$Strong, ?, ?\rangle$                          | $Warm\neq Cool,Same\neq Change$. $h$ is not consistent with $x_4$ at positions $Water$ and $Forcast$. |

# 4.3 List-Then-Eliminate Algorithm
## 4.3.1 Consistency and Version Space
- [DEF] A hypothesis $h$ is ***consistent*** with a set of training examples $D$ of target concept $c$ iff:
	- $Consistent(h,D)\equiv\forall\langle x,c(x)\rangle\in D, h(x)=c(x)$
	- $h$ satisfied all the input-output pairs in $D$.
- [DEF] The ***version space*** $VS_{H,D}$ with respect to:
	- Hypothesis space $H=\{h_1, h_2, ...\}$
	- Training examples $D$
- is defined as:
	- $VS_{H,D} \equiv \{h\in H|Consistent(h,D)\}$
	- A subset from the set of al hypotheses $H$ that's consistent with all training examples in $D$.

## 4.3.2 Representation of Version Spaces
- [DEF] **Specific Boundary** ($S$) of $VS_{H,D}$
	- The set of $VS_{H,D}$'s maximally **specific** members of $H$ that's consistent with $D$.
	- $S\equiv \{s\in H | Consistent(s,D)$
	  $\land (\neg\exists s'\in H)[(s>_g s') \land Consistent(s',D)]\}$ 
		- $s$ is consistent with $D$.
		- Doesn't $s'$ who consistent with $D$ and also less general than the $s$ in $S$.
- [DEF] **General Boundary** ($G$) of $VS_{H,D}$
	- The set of $VS_{H,D}$'s maximally **general** members of $H$ that's consistent with $D$.
	- $G\equiv \{g\in H | Consistent(s,D)$
	  $\land (\neg\exists g'\in H)[(g'>_g g) \land Consistent(s',D)]\}$ 

## 4.3.3 List-Then-Eliminate Algorithm
Procedure:
- For each training example $\langle x,c(x)\rangle\in D$ do:
	- if$c(x)=1$ do:
		- Remove from $G$ any hypothesis inconsistent with $d$.
		- For each hypothesis $s\in S$ that's inconsistent with $d$:
			- Remove $s$ from $S$.
			- Add to $S$ a small minimal generalizations $h$ of $s$ s.t.,
				- $h$ is consistent with $d$, and some member of $G$ is more general than $h$.
			- Remove from $S$ any hypothesis that 
				- is more general than another hypothesis in $S$.
	- if$c(x)=0$ do:

**Example**

| Training Samples                                                     | S                                                                               | G                                                                                                                                                                                                                                                                                                                                                     | Description                                                                                                                                                     |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                                                      | $S_0=\langle\emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset\rangle$ | $G_0=\langle?,?,?,?,?,?\rangle$                                                                                                                                                                                                                                                                                                                       |                                                                                                                                                                 |
| $x_1=\langle Sunny, Warm, Normal,$<br>$Strong, Warm, Same\rangle, +$ | $S_1=\langle Sunny, Warm, Normal,$<br>$Strong, Warm, Same\rangle$               | $G_1=G_0$                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                 |
| $x_2=\langle Sunny, Warm, High,$<br>$Strong, Warm, Same\rangle, +$   | $S_2=\langle Sunny, Warm, ?,$<br>$Strong, Warm, Same\rangle$                    | $G_2=G_1$                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                 |
| $x_3=\langle Rainy, Cold, High,$<br>$Strong, Warm, Change\rangle, -$ | $S_3=S_2$                                                                       | $G_3^{before}=$<br>$\langle Sunny,?,?,?,?\rangle$<br>$\langle ?,Warm,?,?,?,?\rangle$<br>$\langle ?,?,?,?,?\rangle$<br>$\langle ?,?,?,Strong,?\rangle$<br>$\langle ?,?,?,Warm,?\rangle$<br>$\langle ?,?,?,?,Same\rangle$<br><br>$G_3^{after}=$<br>$\langle Sunny,?,?,?,?,?\rangle$<br>$\langle ?,Warm,?,?,?,?\rangle$<br>$\langle ?,?,?,?,Same\rangle$ | Remove inconsistent hypothesis. <br><br>Here, the classification is "-". Therefore, to be "consistent", the target value should be different from the original. |
| $x_4=\langle Sunny, Warm, High,$<br>$Strong, Cool, Change\rangle, +$ | $S_4=\langle Sunny, Warm, ?,$<br>$Strong, ?, ?\rangle$                          | $G_4=$<br>$\langle Sunny,?,?,?,?,?\rangle$<br>$\langle ?,Warm,?,?,?,?\rangle$                                                                                                                                                                                                                                                                         | $Same\neq Change$, inconsistent.                                                                                                                                |
