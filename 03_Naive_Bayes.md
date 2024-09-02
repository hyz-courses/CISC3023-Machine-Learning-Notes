# 3.0 Why Naive Bayes?
- Uncertainty
	- Can't conclude something with 100% confidence
- Weak Implications
	- Hard to establish concrete correlations between IF and THEN.
	- Handle vague associations.
- Imprecise Language
	- Natural language is ambiguous.
	- We describe facts with: sometimes, often, frequently, hardly, ....
	- Difficult to establish IF-THEN rules based on NL.

# 3.1 Basic Probability Theory
## 3.1.1 [DEF] Probability
- The probability of an event 
	- = the proportion of cases in which the event occurs.
	- Expression: From 0 (absolute impossible) -> Unity (Absolute certain)
	- Mostly strictly between 0 and 1. Each event has at least two outcomes: Success or failure.
		- $P(success)=\dfrac{s}{s+f}$
		- $P(failure)=\dfrac{f}{s+f}$
## 3.1.2 [DEF] Conditional Probability
- Let: A, B: Event
- Conditional Probability:
	- The probability that: If B occur, then A occur.
	- $P(A|B)=\dfrac{num(A and B occur)}{num(B occur)}$
	- $P(A|B)=\dfrac{P(A\cap B)}{P(B)}=\dfrac{P(B|A)P(A)}{P(B)}$ (yields the Bayesian Rule)

# 3.2 Bayesian Reasoning
## 3.2.1 Baysian Rule:
- Given: Event E (Evidence)
- Get: The prob. that event H (Hypothesis) will occur, as P.
	- $P(H|E)=\dfrac{P(E|H)P(H)}{P(E)}=\dfrac{P(E|H)P(H)}{P(E|H)P(H)+P(E|\neg H)P(\neg H)}$

## 3.2.2 Variances:
- Single Evidence, Multiple Hypothesis:
	- $P(H_i|E)=\dfrac{P(E|H_i)P(H_i)}{\sum_{k=1}^{m}{P(E|H_k)P(H_k)}}$
- Multiple Evidence, Multiple Hypothesis:
	- $P(H_i|E_1,E_2,...,E_n)=\dfrac{P(E_1,E_2,...,E_n)P(H_i)}{\sum_{k=1}^{m}{P(E_1,E_2,...,E_n|H_k)P(H_k)}}$
	- $\approx\dfrac{P(E_1|H_i)\times P(E_2|H_i)\times ... \times P(E_n|H_i)}{\sum_{k=1}^{m}{[P(E_1|H_k)\times P(E_2|H_k)\times ...\times P(E_n|H_k)\times P(H_k)}]}$, if conditional independence holds.

## Example
- Given the prior an conditional probs as follows:

|                   | $H_1$  | $H_2$  | $H_3$  |
| ----------------- | ------ | ------ | ------ |
| $P(H_i)$          | $0.40$ | $0.35$ | $0.25$ |
| $P(E_1\vert H_i)$ | $0.3$  | $0.8$  | $0.5$  |
| $P(E_2\vert H_i)$ | $0.9$  | $0.0$  | $0.7$  |
| $P(E_3\vert H_i)$ | $0.6$  | $0.7$  | $0.9$  |
- Want $P(H_1|E_3)$.
- $P(H_3|E_3)=\dfrac{P(E_3|H_3)P(H_3)}{P(E_3)}$
	- $P(E_3|H_3)P(H_3)=0.9 \times 0.25=0.36$
	- $P(E_3)=P(E_3|H_1)P(H_1)\times P(E_3|H_2)P(H_2)\times P(E_3|H_3)P(H_3)$
		- $=0.6\times 0.4+0.7\times 0.35+0.9\times 0.25=0.2838$
		- 

# 3.3 Naive Bayes Classifier
## 3.3.1 Maximum A Posteriori
- $H_{conclusion}=argmax_{h\in H}P(h|E)$
	- $=argmax_{h\in H}\dfrac{P(E|h)P(h)}{P(E)}$
	- $=argmax_{h\in H}P(E|h)P(h)$
- Omit the $P(E)$ since it's constant, which is independent from the hypothesis.

## 3.3.2 Naive Bayes Estimation
- Given:
	- A conjunctive test sample: $x_1,x_2,...,x_n$
- $c_{MAP}=argmax_{c_j\in C}P(c_j|x_1,x_2,...x_n)$
	- $=argmax_{c_j\in C}\dfrac{P(x_1,x_2,...,x_n|c_j)P(c_j)}{P(x_1,x_2,...,x_n)}$
	- $=argmax_{c_j\in C}P(x_1,x_2,...,x_n|c_j)P(c_j)$
	- $=argmax_{c_j\in C}[P(x_1|c_j)\times P(x_2|c_j)\times ... \times P(x_n|c_j)]\times P(c_j)$
	- $=argmax_{c_j\in C}P(c_j)\times \prod_{k=1}^{n}P(x_k|c_j)$

## Example
| Day | Outlook  | Temp | Humitity |  Wind  | Play Tennis |
| :-: | :------: | :--: | :------: | :----: | :---------: |
|  1  |  Sunny   | Hot  |   High   |  Weak  |   **No**    |
|  2  |  Sunny   | Hot  |   High   | Strong |   **No**    |
|  3  | Overcast | Hot  |   High   |  Weak  |   **Yes**   |
|  4  |   Rain   | Mild |   High   |  Weak  |   **Yes**   |
|  5  |   Rain   | Cool |  Normal  |  Weak  |   **Yes**   |
|  6  |   Rain   | Cool |  Normal  | Strong |   **No**    |
|  7  | Overcast | Cool |  Normal  | Strong |   **Yes**   |
|  8  |  Sunny   | Mild |   High   |  Weak  |   **No**    |
|  9  |  Sunny   | Cool |  Normal  |  Weak  |   **Yes**   |
| 10  |   Rain   | Mild |  Normal  |  Weak  |   **Yes**   |
| 11  |  Sunny   | Mild |  Normal  | Strong |   **Yes**   |
| 12  | Overcast | Mild |   High   | Strong |   **Yes**   |
| 13  | Overcast | Hot  |  Normal  |  Weak  |   **Yes**   |
| 14  |   Rain   | Mild |   High   | Strong |   **No**    |
Known: Outlook=sunny, Temp=cool, Humidity=high, Wind=strong.
Want: Play tennis or not?

Do:
- $MAP(Yes|sunny, cool, high, strong)$
	- $=P(sunny, cool, high, strong|Yes)\times P(Yes)$
	- $=P(sunny|Yes)\times P(cool|Yes)\times P(high|Yes)\times P(strong|Yes)\times P(Yes)$
	- $=[]\times \dfrac{9}{14}$