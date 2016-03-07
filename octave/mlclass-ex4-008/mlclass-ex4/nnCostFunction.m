function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 x 401
Theta2_grad = zeros(size(Theta2)); % 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% X: 5000 x 401
% Theta1: 25 x 401
% Theta2: 10 x 26

X = [ones(m, 1) X];
Z2 = X * Theta1'; % 5000 x 25
A2 = sigmoid(Z2); % size 5000 x 25
A2 = [ ones(m,1) A2 ]; % size 5000 x 26
Z3 = A2 * Theta2'; % 5000 x 26
A3 = sigmoid(Z3); % 5000 x 10

for i = 1:m
    for k = 1:num_labels
        yk = y(i) == k;
        cost = -yk * log( A3(i,k) ) - (1-yk) * log( 1 - A3(i,k) );
        J = J + cost;
    endfor
endfor
J = J/m;

for t = 1:m
    % Step 1
    yt = y(t);
    a1 = X(t,:);  % size 1 x 401
    z2 = Z2(t,:); % 1 x 25
    % z2 = [1 z2];
    a2 = A2(t,:); % 1 x 26
    z3 = Z3(t,:); % 1 x 10
    a3 = A3(t,:); % 1 x 10
    
    % Step 2
    c = 1:num_labels;
    delta_3 = a3 - (c == yt);
    
    % Step 3
    % Theta2: 10 x 26; delta_3: 1 x 10; z2 is 1 x 25
    % Result: delta_2: 1 x 26
    delta_2 = (delta_3 * Theta2) .* [1 sigmoidGradient(z2)];
    
    % Theta1_grad is 25 x 401
    % delta_2 is 1 x 26
    % a1 is 1 x 401
    Theta1_grad = Theta1_grad + delta_2(2:end)' * a1;
    
    % Theta2_grad: 10 x 26
    % delta_3 is 1 x 10
    % a2 is 1 x 26    
    Theta2_grad = Theta2_grad + delta_3' * a2;
endfor

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularized version of cost function and gradient
J = J + lambda * (sum( Theta1(:,2:end)(:) .^ 2 ) + sum( Theta2(:,2:end)(:) .^ 2 )) / (2*m);

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end) / m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end) / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
