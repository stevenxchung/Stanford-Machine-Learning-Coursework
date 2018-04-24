# Stanford - Machine Learning

In this course we cover the following topics:

* Week 1
	* Introduction
	* Linear Regression with One Variable
	* Linear Algebra Review
* Week 2
	* Linear Regression with Multiple Variables
	* Octave/Matlab Tutorial
* Week 3
	* Logistic Regression
	* Regularization
* Week 4
	* Neural Networks: Representation
* Week 5
	* Neural Networks: Learning
* Week 6
	* Advice for Applying Machine Learning
	* Machine Learning System Design
* Week 7
	* Support Vector Machines
* Week 8
	* Unsupervised Learning
	* Dimensionality Reduction
* Week 9
	* Anomaly Detection
	* Recommender Systems
* Week 10
	* Large Scale Machine Learning
* Week 11
	* Photo Optical Character Recognition

Notes from each section will be appended here as well as available to view in each subfolder for each week.

---

# Machine Learning Week 1

In Week 1 we cover the following topics:
* Linear Regression with One Variable
* Linear Algebra (Review)

## Introduction

Machine Learning
* Grew out of work in AI
* New capability for computers

Examples:
* Database mining: Large datasets from growth of automation/web. E.g., Web click data, medical records, biology, engineering
* Applications can’t program by hand: E.g., Autonomous helicopter, handwriting recognition, most of Natural Language Processing (NLP), Computer Vision
* Self-customizing programs: E.g., Amazon, Netflix product recommendations

What is machine learning?
* Arthur Samuel (1959). Machine Learning: *"Field of
study that gives computers the ability to learn
without being explicitly programmed."*

* Tom Mitchell (1998) Well-posed Learning
Problem: *"A computer program is said to learn
from experience E with respect to some task T
and some performance measure P, if its
performance on T, as measured by P, improves
with experience E."*

Two main types of machine learning algorithms:
* Supervised learning: The majority of practical machine learning problems. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected. Learning stops when the algorithm achieves an acceptable level of performance. We can group supervised learning problems into:
	* Regression problems: When the output variable is a real value e.g., "dollars" or "weight".
	* Classification problems: When the output variable is a category e.g., "black" or "white".

* Unsupervised learning: When there exists input data and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. We can group unsupervised learning problems into:
	* Clustering problems: When we want to discover the inherent groupings in the data e.g., grouping customers by purchasing behavior.
	* Association problems: When we want to discover rules that describe large portions of our data e.g., people that buy X also tend to buy Y.

<div align="center">
	<img src="Week 1/photos/sup-vs-unsup.png">
	<h3>Figure 1-1. Supervised learning versus unsupervised learning</h3>
</div>

## Linear Regression with One Variable

In a linear regression problem we predict a real-valued output using existing data that is the "right answer" for each example in the data. Furthermore, we can represent the data with a hypothesis function and use a cost function (squared error function) to minimize the parameters of the hypothesis.

<div align="center">
	<img src="Week 1/photos/linearreg.jpg">
	<h3>Figure 1-2. Example of linear regression and hypothesis</h3>
</div>

<div align="center">
	<img src="Week 1/photos/costfunctionsetup.jpg">
	<h3>Figure 1-3. Cost function setup</h3>
</div>

A way of minimizing the cost function is to use the gradient descent method. Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).

<div align="center">
	<img src="Week 1/photos/gradientdescent.jpg">
	<h3>Figure 1-4. Example of gradient descent</h3>
</div>

## Linear Algebra (Review)

Matrices and vectors
* Dimensions of matrix: rows x columns

<div align="center">
	<img src="Week 1/photos/matrix-operation-formula-algebra.jpg">
	<h3>Figure 1-5. Matrix operations sheet</h3>
</div>

---

# Machine Learning Week 2

In Week 2 we cover the following topics:
* Linear Regression with Multiple Variables
* Octave/Matlab Tutorial

## Linear Regression with Multiple Variables

Linear regression with multiple variables is not too different than linear regression for single variables. When dealing with multiple variables, we must change the hypothesis, cost function, and gradient descent algorithm to represent multiple features. One way to do this is by using vector notation. An example of linear regression with multiple variables setup is denoted in Figure 2-1.

<div align="center">
	<img src="Week 2/photos/linearregwmultivar.jpg">
	<h3>Figure 2-1. Linear regression with multiple variables setup</h3>
</div>

Feature scaling: make sure features are on a similar scale. One technique used to accomplish this is by using [mean normalization](https://en.wikipedia.org/wiki/Normalization_(statistics)). Feature scaling could drastically change the appearance of a graph and therefore, drastically improve the precision accuracy of a model (see Figure 2-2 for impacts of feature scaling).

<div align="center">
	<img src="Week 2/photos/featurescale.jpg">
	<h3>Figure 2-1. Unscaled results versus scaled results</h3>
</div>

Learning rate: Specified step-size the algorithm (in this case the gradient descent algorithm) takes at each iteration when trying to minimize the cost function.

In general, if the learning rate:
* Is too small, the convergence could be expected to happen slower than normal
* Is too large, the cost function may not decrease on every iteration and may not converge

Other regression techniques:
* Polynomial regression: Differs from linear regression with the addition of an nth degree term.
* Normal equation: Method to solve for the function parameters (commonly denoted by theta) analytically.

<div align="center">
	<img src="Week 2/photos/gradvsnorm.jpg">
	<h3>Figure 2-3. Gradient descent versus normal equation</h3>
</div>

---

# Machine Learning Week 3

In Week 3 we cover the following topics:
* Logistic Regression
* Regularization

## Logistic Regression

Logistic regression is intended for binary (two-class) classification problems. It will predict the probability of an instance belonging to the default class, which can be snapped into a 0 or 1 classification.

A logistic regression model could be represented as follows:

<div align="center">
	<img src="Week 3/photos/logregmodel.jpg">
	<h3>Figure 3-1. Logistic regression model</h3>
</div>

We can use a decision boundary to make predictions:

<div align="center">
	<img src="Week 3/photos/dbound.jpg">
	<h3>Figure 3-2. Logistic regression decision boundary setup</h3>
</div>

Additionally, the cost function and gradient descent algorithm for logistic regression looks fairly similar to that of linear regression:

<div align="center">
	<img src="Week 3/photos/costandgrad.jpg">
	<h3>Figure 3-3. Logistic regression cost function and gradient descent</h3>
</div>

In addition to linear and logistic regression techniques, we can also use optimization algorithms such as conjugate gradient, BFGS, L-BFGS, etc. The advantages of using optimization algorithms is that we do not manually have to pick the step-size variable. Optimization algorithms tend to be faster than gradient descent.

Multi-class classification: When there is more than one class, we can apply the one-vs-all method:

<div align="center">
	<img src="Week 3/photos/onevsall.jpg">
	<h3>Figure 3-4. Multi-class classification: one-vs-all method</h3>
</div>

## Regularization

Overfitting: If we have too many features, the learned hypothesis may fit the training set very well, but fail to generalize to new examples (predict prices on new examples).

To address overfitting we can do the following:
* Reduce the number of features
	* Manually select which features to keep
	* Model selection algorithm
* Regularization
	* Keep all the features, but reduce magnitude/values of parameters
	* Works well when we have a lot of features, each of which contributes a bit to predicting the output.

The cost function for regularization incudes an extra parameter to the cost function:

<div align="center">
	<img src="Week 3/photos/regular.jpg">
	<h3>Figure 3-5. Regularization</h3>
</div>

We do not want the lambda setting to be too small or else it will result in overfitting. Conversely, we do not want the lambda setting to be too large or else minimization of the cost function will result in underfitting.

The regularization parameter can be added to a linear or a logistic cost function as well as a gradient descent.

---

# Machine Learning Week 4

In Week 4 we cover the following topics:
* Neural Networks: Representation

## Neural Networks: Representation

Neural networks originated from algorithms that try to mimic the brain. The use of such algorithms became popular in the 80s and early 90s but soon diminished in the late 90s. Today, neural networks are being used for many applications.

A neural network could be represented as follows:

<div align="center">
	<img src="Week 4/photos/neuralnet.jpg">
	<h3>Figure 4-1. Neural network model</h3>
</div>

Where the activation terms denoted by "a" occur in layer 2 to layer n (hidden layer). Alternatively, neural networks can be [vectorized](http://ufldl.stanford.edu/wiki/index.php/Neural_Network_Vectorization).

When trying to classify more than one output units a representation model could look like this:

<div align="center">
	<img src="Week 4/photos/onevsall.jpg">
	<h3>Figure 4-2. Neural network model: one-vs-all</h3>
</div>

# Machine Learning Week 5

In Week 5 we cover the following topics:
* Neural Networks: Learning

## Neural Networks: Learning

A neural network classification problem could be represented as either a binary or multi-class classification:

<div align="center">
	<img src="Week 5/photos/neuralnetclass.jpg">
	<h3>Figure 5-1. Neural network binary and multi-class classification</h3>
</div>

The cost function for a neural network could be represented as:

<div align="center">
  <img src="Week 5/photos/costfunc.jpg">
  <h3>Figure 5-2. Neural network cost function</h3>
</div>

Which turns out to be quite similar to the logistic regression cost function.

The gradient computation takes in the same idea as before: we seek to minimize the cost function. Figure 5-3 depicts the general equation needed to compute the gradient descent of a neural network and minimize the cost function:

<div align="center">
  <img src="Week 5/photos/neuralgrad.jpg">
  <h3>Figure 5-3. Gradient descent for a neural network</h3>
</div>

Backpropagation is at the core of how neural networks learn.

To include [backpropagation](https://brilliant.org/wiki/backpropagation/) we would first need to have our neural network defined and an [error function](https://brilliant.org/wiki/artificial-neural-network/#training-the-model). In backpropagation, the final weights are calculated first and the first weights are calculated last. This allows for the efficient computation of the gradient at each layer.

In general, we forward propagate to get the output and compare it with the real value to get the error. Then, we backpropagate to minimize the error (find the derivative of error with respect to each weight then subtract this value from the weight value). This process repeats and continues until we reach a minima for error value.

To summarize on how to train a neural network:

<div align="center">
  <img src="Week 5/photos/steps.jpg">
  <h3>Figure 5-4. Training a neural network</h3>
</div>

In addition to the steps proposed in Figure 5-4:
* Use gradient checking to compare the partial derivative of the cost function with respect to the weights computed using backpropagation versus using numerical estimate of gradient of the cost function
* Disable the gradient checking code
* Use gradient descent or advanced optimization method with backpropagation to try to minimize the cost function as a function of the weights
