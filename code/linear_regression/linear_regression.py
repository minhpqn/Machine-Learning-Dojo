"""
Implement batch gradient descent algorithm with python
Use the implemented algorithm to fit parameters \theta from dataset
"""

import sys
import numpy as np
import pandas as pd

def compute_cost(X, y, theta):
	m = y.shape[0]
	return sum( ( np.sum(X * theta, axis=1) - y ) ** 2 ) / (2*m)

def gradient_descent(X, y, theta, alpha, num_iters):
	m = y.shape[0]
	J_history = np.zeros(num_iters);

	for iter in xrange(iterations):
		sum1 = 0
		sum2 = 0
		for i in xrange(m):
			tmp = sum( theta * X[i,:] ) - y[i]
			sum1 += tmp
			sum2 += tmp * X[i,1]

		theta[0] -= alpha * sum1/m;
		theta[1] -= alpha * sum2/m

		# Save the cost J in every iteration    
		J_history[iter] = compute_cost(X, y, theta);

	return (theta, J_history)

if __name__ == '__main__':
	df = pd.read_csv('./ex1data1.txt', header=None)

	iterations = 1500;
	alpha = 0.01;
	
	theta = np.zeros( 2 );

	X = df.ix[:,0].as_matrix()
	y = df.ix[:,1].as_matrix()

	m = y.shape[0]

	X = X.reshape(m,1)

	X = np.concatenate( ( np.ones( (m, 1) ), X ), axis = 1 )

	print 'Initial cost function: %s' % compute_cost(X, y, theta)

	theta, _ = gradient_descent(X, y, theta, alpha, iterations)

	print 'Theta found by gradient descent: %s' % theta

	# Predict values for population sizes of 35,000 and 70,000
	predict1 = sum( [1, 3.5] *theta ) * 10000;
	print('For population = 35,000, we predict a profit of %f' %
	       predict1);
	predict2 = sum( [1, 7] * theta ) * 10000;
	print('For population = 70,000, we predict a profit of %f' %
		   predict2);

