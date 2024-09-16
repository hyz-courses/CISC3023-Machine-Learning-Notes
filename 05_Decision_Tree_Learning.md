# 5.2 Entropy
## 5.2.1 Attributes & Purity of Decision
- We want to know which attribute is the best classifier.
- Each attribute can be represented as a node, splitting test sample into 2 classes (in the binary case)
- The *purer* the splitting, the better the decision.

## 5.2.2 Entropy
- [DEF] Entropy 熵
	- Given a collection $S$ with portions of both positive and negative $p_+, p_-$, 
	  the entropy of the collection would be:
	- $Entropy(S)=-p_+log_2p_+-p_-log_2p_-$
	- The higher the entropy, the fewer the information the sample contains.
- [Prop] Entropy 熵的性质
	- Domain is $[0,1]$;
	- Monotonically, Increase in $p_+\in[0,\dfrac{1}{2}]$, Decrease in $p_+\in[\dfrac{1}{2},1]$;
	- Reaches the peak at $p=\dfrac{1}{2}$, reaches the lowest point at $p_+=0$ and $p_+=1$.

## 5.2.3 Information Gain
By calculating the difference between
- The entropy **BEFORE** splitting, and
- The entropy **AFTER** splitting,
we can calculate the "Information Gained" from the splitting.

## 5.2.4 From Entropy to Information Gain
Given:
- A decision node, with the amounts in each class:
	- Collection 1 ($S_1$):
		- Positive: $n_+^{1}$
		- Negative: $n_-^{1}$
	- Collection 2 ($S_2$):
		- Positive :$n_+^{2}$
		- Negative: $n_-^{2}$
- Example:
	- $n_+^1=3, n_-^1=4$
	- $n_+^2=6, n_-^2=1$
```mermaid
flowchart TD
A[Humidity (9+, 5-)]--High-->B[3+, 4-]
A[Humidity]--Low-->C[6+, 1-]
```
Do:
- Get the proportion of positive & negative samples in both collections.
	- $S_1$:  $p_+^1=\dfrac{n_+^1}{n_+^1+n_-^1}; \ p_-^1=\dfrac{n_-^1}{n_+^1+n_-^1}$
	- $S_2$: $p_+^2=\dfrac{n_+^2}{n_+^2+n_-^2}; \ p_-^2=\dfrac{n_-^2}{n_+^2+n_-^2}$
- Calculate Entropies:
	- Calculate entropy **AFTER** splitting:
		- $Entropy(S_1)=-p_+^1log_2{p_+^1}-p_-^1log_2{p_-^1}$
		- $Entropy(S_2)=-p_+^2log_2{p_+^2}-p_-^2log_2{p_-^2}$
	- Calculate overall **BEFORE** splitting:
		- $p_+=\dfrac{(p_+^1+p_+^2)}{(p_+^1+p_+^2)+(p_-^1+p_-^2)}$
		- $p_-=\dfrac{(p_-^1+p_-^2)}{(p_+^1+p_+^2)+(p_-^1+p_-^2)}$
		- $Entropy(S)=-p_+log_2{p_+}-p_-log_2{p_-}$
- Calculate Information Gain:
	- $Gain(S)=Entropy(S)-\sum_{i\in\{1,2\}}\dfrac{|S_i|}{|S|}Entropy(S_i)$, where
	- $|S_i|=n_+^i+n_-^i, |S|=\sum_{i\in{1,2}}|S_i|$
