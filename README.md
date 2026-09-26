## Leo Carten - Machine Perception and Cognition HW 1b
#### I did my HW assignment in Python. Source code can be found in the `main` branch inside of `main.py`. Functions are named after there corresponding HW question. E.g., `question_1b()` corresponds to Question 1b in the textbook.

#### My thoughts and notes are also in the source code as comments. If you have any questions, please contact Leo directly. 

#### Below is an output from my program:
```
------------------------------------------------------------------------------------------
Question 1a: Write a procedure to generate random samples according to a normal distribution N(µ,Σ) in d dimensions.
z=[ 0.18634635 -0.50774834]
covar_matrix=[[2.   0.5 ]
 [0.5  0.25]]
A=[[1.41421356 0.        ]
 [0.35355339 0.35355339]]
Az=[ 0.26353354 -0.11363276]
------------------------------------------------------------------------------------------
Question 1b: Write a procedure to calculate the discriminant function (of the form given in Eq. 47) for a given normal distribution and prior probability P(wi).
For this HW question I will be using 2 classes: Salmon and Bass. I will be using 2 inputs: Length and weight. I will be using 'made-up' length and weights for each fish for training data.
Observed salmon length: [24, 26, 27, 28, 29, 30, 32]
Observed salmon weight: [5.5, 7.0, 8.0, 9.0, 10.0, 11.5, 14.0]
Salmon length mean: 28.0
Salmon weight mean: 9.285714285714286
Observed bass length: [12, 13, 14, 15, 16, 17, 18]
Observed bass weight: [0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7]
bass length mean: 15.0
bass weight mean: 1.5999999999999999
Prior probability for salmon: 60.0%
Prior probability for bass: 40.0%
Score for class 'Salmon' based on inputs length=22.0 and weight=4.9: -25.552391231951127
Score for class 'Bass' based on inputs length=22.0 and weight=4.9: -65.00494505866953
Based on scores, we will assign inputs length=22.0 and weight=4.9 to Salmon class.
------------------------------------------------------------------------------------------
Question 1c: Write a procedure to calculate the Euclidean distance between two arbitrary points.
let p=(2,6)
let q=(-4,10)
The euclidean distance between p and q is: 7.211102550927978
------------------------------------------------------------------------------------------
Question 1d: Write a procedure to calculate the Mahalanobis distance between the mean µ and an arbitrary point x, given the covariance matrix Σ.
Observed salmon length: [24, 26, 27, 28, 29, 30, 32]
Observed salmon weight: [5.5, 7.0, 8.0, 9.0, 10.0, 11.5, 14.0]
Salmon length mean: 28.0
Salmon weight mean: 9.285714285714286
New fish (x) beeing seen: length=24, weight=10
The new fish mahalanobis distance is: 15.737428845483858
------------------------------------------------------------------------------------------
```