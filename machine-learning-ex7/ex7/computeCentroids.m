function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for iter=1:K
    tmp = idx;
    tmp2 = X;
    % change the value of the members of the current cluster to zero
    tmp2(tmp~=iter,:)=0;
    tmp = sum(tmp2~=0);
    % calculate the means
    tmp2 = arrayfun(@(n) sum(tmp2(:,n))/tmp(1) , 1:size(tmp2,2));
    
    % assign the mean
    centroids(iter,:)= tmp2;
end


% =============================================================


end

