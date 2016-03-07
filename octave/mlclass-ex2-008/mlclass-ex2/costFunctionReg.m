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

%for i = 1:m
%    hx = sigmoid(theta' * X(i,:)');
%    J = J - y(i)*log(hx) - (1-y(i))*log(1-hx);
%endfor
%J = J/m;

%for i = 2:size(theta)
%    J = J + lambda/(2*m)*theta(i)^2;
% endfor

hx = sigmoid(X * theta);
J = 1/m * sum( -y .* log(hx) - (1-y) .* log(1 - hx) );
J = J + lambda * sum( theta(2:end) .^ 2 )/(2*m);

% Compute gradient vectors
grad = 1/m * X' * (hx - y);
temp = theta;
temp(1) = 0;
grad = grad + lambda * temp / m;

%for j = 1:size(theta)
%    for i = 1:m
%        hx = sigmoid(theta' * X(i,:)');
%        grad(j) = grad(j) + (hx - y(i)) * X(i,j);
%    endfor
%    grad(j) = grad(j) / m;
%    if (j > 1)
%        grad(j) = grad(j) + lambda * theta(j)/m;
%    endif
%endfor


% =============================================================

end
