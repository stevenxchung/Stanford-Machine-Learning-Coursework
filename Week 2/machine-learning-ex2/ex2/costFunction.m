function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% h is a 100x1 matrix 
h = sigmoid(X * theta);

% Need to transpose y matrix (100x1) in order for operation to work
J = (1/m) * (-transpose(y) * log(h) - (1 - transpose(y)) * log(1 - h))

% Need to transpose (h - y) matrix (100x1) in order for operation to work
% Transpose of final product yields a 3x1 which is what we are looking for
grad = (1/m) * transpose(transpose(h - y) * X)

% =============================================================

end
