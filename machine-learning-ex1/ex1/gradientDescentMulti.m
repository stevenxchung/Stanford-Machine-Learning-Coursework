function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % Subtract y from each element in the 47x1 matrix 
    delta = (X * theta) - y;
    
    % From the inside we have a 1x47 matrix (after transpose) multiplied by
    % a 47x3 matrix which yields a 3x1 matrix after the outer transpose is
    % applied. Then a (1/m) scales this result.
    J = (1/m) * transpose(transpose(delta) * X);
    
    % Theta equation from lecture
    theta = theta - alpha * J;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
