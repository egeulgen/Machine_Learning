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

% empty matrix for later use
mat = zeros(size(X,1),K);

% iterating over the number of desired clusters, notated as K

for clus_idx = 1:K
    % calculating the eucledian distance from each centroid for all samples
    temp = centroids(clus_idx,:);
    
    temp = bsxfun(@minus,X,temp);
    
    % from stackoverflow
    temp = arrayfun(@(n) norm(temp(n,:)), 1:size(temp,1));
    temp = temp';
    
    % appending current distance to the matrix
    mat(:,clus_idx) = temp;
end

temp = arrayfun(@(n) min(mat(n,:)), 1:size(mat,1));

idx = arrayfun(@(n) find(mat(n,:)==temp(n)), 1:size(mat,1));

% =============================================================

end

