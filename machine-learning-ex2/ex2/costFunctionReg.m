function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% h is a 100x1 matrix 
h = sigmoid(X * theta);

% Need to transpose y matrix (100x1) in order for operation to work and
% adjust theta to include theta(1)
J = (1/m) * (-transpose(y) * log(h) - (1 - transpose(y)) * log(1 - h)) + ...
    (lambda/(2*m)) * (transpose(theta) * theta - theta(1)^2); 

% Similar to before but we need to account for the extra zero, so initTheta
% will have 27 ones with a zero at initTheta(1)
initTheta = ones(size(theta));
initTheta(1) = 0;

% Almost the same as without regularization, just add lambda and thetas
grad = (1/m) * transpose(transpose(h - y) * X) + (lambda/m) * ...
    (theta.*initTheta);

% =============================================================

end
