function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the a1 data matrix taken from predictOneVsAll.m, otherwise
% a1 will be a 5000x400 matrix
a1 = [ones(m, 1) X];

% Here z2 is a 5000x25 matrix which will be used to determine a2
z2 = a1 * transpose(Theta1);
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];

% Repeat process with next unit, this is directly from lecture
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3)

% Same as predictOneVsAll where we use the max() function to obtain max
% values
[var, p] = max(a3, [], 2);

% =========================================================================


end
