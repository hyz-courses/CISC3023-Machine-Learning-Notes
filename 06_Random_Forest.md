# 6.0 Why RF?
- Single decision tree does not perform well, but it is super fast.
- Random forest is an ensemble classifier that 
	- combines many decision trees, 
	- outputs the class that is the mode of the class's output by individual trees

# 6.1 Algorithm
Given:
- $N$: The number of training data.
- $M$: The number of variables in each data.
Do:
- For each tree:
	- Boot strap to get $n$ data from $N$ training data.
	- For each node:
		- Randomly select $m$ feature variables, choose one that makes the purest decision.

Each tree is fully grown and not pruned.