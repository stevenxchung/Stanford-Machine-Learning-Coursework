function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% K specifies the number of classes in Y which is a vector of y's
K = num_labels;
eyeY = eye(K);
Y = eyeY(y, :);

%% Part 1: Feedforward the neural network and return the cost variable J

% Build a1, a2, and a3 based on previous lectures
a1 = [ones(m, 1), X];
z2 = a1 * transpose(Theta1);
a2 = [ones(size(sigmoid(z2), 1), 1), sigmoid(z2)];
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);

% Implement logistic cost function with multi-summation from lecture
J = 1/m * sum(sum((-Y.*log(a3)) - ((1 - Y).*log(1 - a3)), 2));

% Regularization, as mentioned in lecture we use the non biased thetas to
% calculate the regularized J
nonbiasedTheta1 = Theta1(:, 2:end);
nonbiasedTheta2 = Theta2(:, 2:end);

% From ex4.pdf we can implement the new J which is regularized
J = J + lambda/(2*m) * (sum(sumsqr(nonbiasedTheta1)) +...
    sum(sumsqr(nonbiasedTheta2)));

%% Part 2: Implement the backpropagation algorithm to compute the gradients
% Theta1_grad and Theta2_grad

% Could not get for loop to work so tried to find another method
%{
% Following the instructions from ex4.pdf we have initial delta values
initDelta1 = 0;
initDelta2 = 0;

for t = 1:m
    
    % Setting the input layer's values a1 to the t-th training example xt,
    % also need to add input bias to a1 and a2, this step is almost
    % identical to a1, a2, and a3 in Part 1
    a1 = [1, transpose(X(t, :))];
    z2 = a1 * transpose(Theta1);
    a2 = [1, sigmoid(z2)];
    z3 = a2 * transpose(Theta2);
    a3 = sigmoid(z3);
    
    % Each output unit k in layer 3 needs to be set
    Delta3 = a3 - transpose(Y(t, :));
    
    % For the hidden layer l = 2
    Delta2 = (transpose(nonbiasedTheta2) * Delta3).*sigmoidGradient(z2);
    
    % Accumulate the gradient, update initDelta1 and initDelta2
    initDelta1 = initDelta1 + (Delta2 * transpose(a1));
    initDelta2 = initDelta2 + (Delta3 * transpose(a2));
    
end

%}

% Delta2 and Delta3 are required to be the same dimensions as a2 and a3
Delta3 = a3 - Y;
Delta2 = (Delta3 * Theta2).*[ones(size(z2, 1), 1) sigmoidGradient(z2)];

% bigDelta1 and bigDelta2 are required to be the same dimensions as Theta1
% and Theta2
bigDelta1 = transpose(Delta2(:, 2:end)) * a1;
bigDelta2 = transpose(Delta3) * a2;

% Unregularized gradient for the neural network cost function
Theta1_grad = Theta1_grad + 1/m * bigDelta1;
Theta2_grad = Theta2_grad + 1/m * bigDelta2;

%% Part 3: Implement regularization with the cost function and gradients

% Regularization based on ex4.pdf regularized neural networks, use 2:end to
% ignore the first column since we do not want to use the bias term
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * nonbiasedTheta1;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * nonbiasedTheta2;
    
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
