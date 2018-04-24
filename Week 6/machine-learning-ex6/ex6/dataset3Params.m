function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Initial conditions taken from section 1.2.3 Example Dataset 3
minimum = inf;
paramVec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% To find the best C and sigma values, use a nested for loop
% For each C value in the parameter vector
for CValue = paramVec,
    % For each sigma value in the parameter vector
    for sigmaValue = paramVec,
        fprintf('Selection: [CValue, sigmaValue] = [%f %f]\n',...
        CValue, sigmaValue);
        % Call training function
        trainingSVM = svmTrain(X, y, CValue, @(x1, x2) gaussianKernel(...
            x1, x2, sigmaValue));
        % Call predict function
        predictSVM = svmPredict(trainingSVM, Xval);
        % From section 1.2.3 in ex6.pdf
        errorSVM = mean(double(predictSVM ~= yval));
        fprintf('Prediction error: %f\n', errorSVM);
        if errorSVM <= minimum,
            C = CValue;
            sigma = sigmaValue;
            minimum = errorSVM;
            fprintf('[C, sigma] = [%f %f]\n', C, sigma);
        end
    end
end

fprintf('\nOptimal values: [%f %f] with Prediction error: %f\n\n', C,...
    sigma, minimum);

% =========================================================================

end
