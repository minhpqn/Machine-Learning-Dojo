"""
Implement linear regression for multi variables
"""

import numpy as np
import matplotlib.pyplot as plt

# Compute cost function in the case of multi variables
def compute_cost_multi(X, y, theta):
	m = y.shape[0]
	return  ( ( np.dot(X, theta) - y ) ** 2 ).sum()

def gradient_descent_multi(X, y, theta, alpha, num_iters):
	m = y.shape[0]
	J_history = np.zeros(num_iters)

	for iter in xrange(num_iters):
		# Compute the vector delta
		delta = np.zeros( (X.shape[1],1) )
		for i in xrange( X.shape[1] ):
		 	delta[i] =  ( (np.dot(X, theta) - y) * X[:,[i]] ).sum() / m

		theta = theta - alpha * delta
		J_history[iter] = compute_cost_multi(X, y, theta)

	return theta, J_history

def normal_eqn(X, y):
	theta = np.zeros( (X.shape[0], 1) )
	theta = np.dot( np.dot( np.linalg.pinv( np.dot(X.T, X) ), X.T ), y )

	return theta

# Feature normalization
# Return normalized features, mu, and sigma
def feature_normalize(X):
	X_norm = X
	mu = X.mean(axis=0)
	sigma = X.std(axis=0, ddof=1)

	for i in xrange(X.shape[1]):
		X_norm[:,[i]] = ( X_norm[:,[i]] - mu[i] ) / sigma[i]

	return X_norm, mu, sigma

if __name__ == '__main__':
	data = np.loadtxt('ex1data2.txt', delimiter=',')
	X = data[:,0:2]
	y = data[:,[2]]
	m = y.shape[0]

	print 'First 10 examples of the data set'
	for x_, y_ in zip(X[0:9], y[0:9]):
		print ' x = [%.0f %.0f] y = %.0f' % ( x_[0], x_[1], y_)

	# Feature normalization
	print 'Normalizing features'
	X, mu, sigma = feature_normalize(X)

	# Add ones to X
	X = np.concatenate( ( np.ones( (m, 1) ), X ), axis = 1 )

	# initialize theta, alpha, and num_iters
	alpha = 0.01;
	num_iters = 400;

	# Init Theta and Run Gradient Descent 
	theta = np.zeros( (X.shape[1], 1) );

	print 'Initial cost function: %f' % compute_cost_multi(X, y, theta)

	theta, J_history = gradient_descent_multi(X, y, theta, 
		                                      alpha, num_iters);

	# Plot the convergence graph
	plt.plot( range(1, J_history.shape[0] + 1), J_history )
	plt.xlabel('Number of iterations');
	plt.ylabel('Cost J');
	plt.show()

	print 'Theta computed from gradient descent: %s' % theta.T

	# Estimate the price of a 1650 sq-ft, 3 br house
	# Recall that the first column of X is all-ones. Thus, it does
	# not need to be normalized.
	price = 0; 
	print mu
	print sigma

	unseen_data = np.array( [1650, 3], dtype=np.float64 );
	unseen_data[0] = (unseen_data[0] - mu[0])/sigma[0];
	unseen_data[1] = (unseen_data[1] - mu[1])/sigma[1];
	unseen_data = np.concatenate( (np.ones(1), unseen_data) );
	price = (theta.T * unseen_data).sum();

	print( 'Predicted price of a 1650 sq-ft, 3 br house ' 
	       '(using gradient descent): $%f\n' % price);

	# Normal Equations
	print 'Solving with Normal Equations'
	data = np.loadtxt('ex1data2.txt', delimiter=',')
	X = data[:,0:2]
	y = data[:,[2]]
	m = y.shape[0]

	X = np.concatenate( ( np.ones( (m, 1) ), X ), axis = 1 )

	theta = normal_eqn(X, y)
	print 'Theta computed from Normal Equations: %s' % theta.T

	x = np.array( [1, 1650, 3 ], dtype=np.float64 )
	price = (theta.T * x).sum();

	print( 'Predicted price of a 1650 sq-ft, 3 br house ' 
	       '(using normal quations): $%f\n' % price);








