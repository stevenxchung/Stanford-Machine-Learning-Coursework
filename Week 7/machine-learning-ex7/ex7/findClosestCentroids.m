function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% First loop though length of rows in X
for i = 1:size(X, 1)
    % Initialize empty x
    x = [];
    % Looping through length of rows in centroids
    for j = 1:K
        % Build array by taking all elements from each row of X and adding
        % it up until x = X, do this 3 times (K = 3)
        x = [x; X(i, :)];
    end
    % This is directly from 1.1.1 Finding closest centroids
    displace = sum((x - centroids).^2, 2);
    [max, idx(i)] = min(displace);
end

% =============================================================

end

