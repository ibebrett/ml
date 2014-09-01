function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% You need to return the following variables correctly 
J = 1/(2*m)*(X*theta - y)'*(X*theta-y);
J += lambda/(2*m)*theta(2:end)'*theta(2:end);

grad = zeros(size(theta));
for j = 1:m
    grad += (X(j, :)*theta - y(j))*X(j, :)';
end
grad = (1/m)*grad;
theta_copy = theta(:);
theta_copy(1) = 0;
grad += (lambda/m)*theta_copy;
grad = grad(:);

end
