function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n= size(X,2); % number of features
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

temp = 0;

for i = 1:m
    h = sigmoid(theta'*X(i,:)');
    temp = temp - y(i)*log(h) - (1-y(i))*log(1-h);
end
J = 1/m * temp;

temp = 0;
for i = 2:n
    temp=temp+theta(i)^2;
end

J = J + (lambda/(2*m))*temp;


for j = 1:length(grad)
    for i = 1:m
        h = sigmoid(theta'*X(i,:)');
        grad(j) = grad(j) + X(i,j)*(h-y(i));
    end
    if j ~= 1
        grad(j) = grad(j) + lambda*theta(j);
    end
    
    grad(j) = 1/m * grad(j);
end




% =============================================================

end
