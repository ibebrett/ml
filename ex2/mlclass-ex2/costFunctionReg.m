function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
J_add = 0;
grad = zeros(size(theta));

J_add = (lambda/(2*m))*(theta(2:end)'*theta(2:end));
for i=1:m
    h = sigmoid(X(i, :)*theta);
    %disp(X(i, :))
    %disp(theta)
    J       += (-(y(i)*log(h) ) - (1-y(i))*log(1-h));
    grad    += (h - y(i))*X(i, :)';
end


J    = (1/m)*J + J_add;
grad = (1/m)*grad;

for i = 2:size(theta)
    grad(i) += (lambda/m)*theta(i);
end

end



