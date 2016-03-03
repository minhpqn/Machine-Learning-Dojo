# -*- coding: utf8 -*-
""" Reproduce programming exercise 5 of machine learning course
    on Coursera using python programming language
    Machine Learning Online Class
    Exercise 5 | Regularized Linear Regression and Bias-Variance
    In this exercise, you will implement regularized linear regression 
    and use it to study models with diffrent bias-variance properties.
"""

import os
import sys
import time
import numpy as np
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt

def linearRegCostFunction(theta, X, y, _lambda):
	""" Regularized linear regression cost function

	Parameters
	--------------
	X : numpy.ndarray
	    Data matrix (each row represents a feature vector)
	y : numpy.ndarray
	    Vector of values corresponding to examples
	theta : numpy.ndarray
	    parameters of linear regression
	_lambda: numpy.float64
	    Regularization parameter

	Returns
	--------------
	cost : numpy.float64
	       Value of regularized linear regression cost function

	Notes
	--------------
	In this implementation, I try to use numpy.ndarray in 2D dimensions
	"""

	m = y.size
	theta2D = theta.reshape( (theta.shape[0],1) )

	return ( ((np.dot(X, theta2D) - y) ** 2).sum()/(2*m) 
		    + _lambda * (theta2D[1:,:] ** 2).sum()/(2*m) )

def linearRegGradient(theta, X, y, _lambda):
	""" Regularized linear regression gradients

	Parameters
	--------------
	X : numpy.ndarray
	    Data matrix (each row represents a feature vector)
	y : numpy.ndarray
	    Vector of values corresponding to examples
	theta : numpy.ndarray
	    parameters of linear regression
	_lambda: numpy.float64
	    Regularization parameter

	Returns
	--------------
	grad : numpy.ndarray
	       gradient vectors for regular linear regression

	Notes
	--------------
	In this implementation, I try to use numpy.ndarray in 2D dimensions
	"""

	m = y.size
	theta2D = theta.reshape( (theta.shape[0],1) )

	grad = np.dot( X.T, np.dot(X, theta2D) - y) / m
	grad[1:] += _lambda * theta2D[1:] / m

	return grad.flatten()	

def trainLinearReg(X, y, _lambda, maxiter=200):
	""" Fitting linear regression

	Returns
	-----------
	theta : numpy.ndarray
	        Parameters that minimize the cost function	
	"""
	from scipy.optimize import fmin_l_bfgs_b, fmin_bfgs, fmin_cg

	initial_theta = np.zeros( X.shape[1] )
	theta = fmin_bfgs(linearRegCostFunction, initial_theta,
	                  fprime=linearRegGradient, 
	                  args=(X, y, _lambda),
	                  maxiter=maxiter, disp=False)
	return theta

def learningCurve(X, y, Xval, yval, _lambda):
	""" Compute errors on training and validation set with variable 
	    number of examples, which are neccessary to plot the learning curve
	    Learning curve is an useful to understand about bias/variance
	    of your model

	Parameters
	-------------
	X, y, Xval, yval : numpy.ndarray
	    Training data, training values, Validation data, validation values
	_lambda: float64
	    Regularization parameter

	Returns
	-------------
	error_train, error_val : numpy.ndarray
	    Training and validation errors with different size of training set
	"""

	error_train = np.zeros( X.shape[0] )
	error_val   = np.zeros( X.shape[0] )
	for i in xrange( X.shape[0] ):
		theta = trainLinearReg(X[0:i+1,:], y[0:i+1,:], _lambda)
		error_train[i] = linearRegCostFunction(theta, X[0:i+1,:], 
			                                   y[0:i+1,:], 0)
		error_val[i]   = linearRegCostFunction(theta, Xval, yval, 0)

	return (error_train, error_val)

def polyFeatures(X, p):
	""" Mapping features in X ( size mx1 ) into polynomial features

	Parameters
	--------------
	X : numpy.ndarray
	p : int
	    number of powers

	Returns
	---------------
	X_poly : numpy.ndarray
	    Matrix containing of polynominal features
	    Size: m x p
	"""

	X_poly = np.zeros( (X.size, p) )
	for t in range(1,p+1):
		X_poly[:,[t-1]] = X ** t

	return X_poly

def featureNormalize(X, *args):
	""" Normalize features in X
	
	Returns a normalized version of X where the mean value of 
	each feature is 0 and the standard deviation is 1. 
	This is often a good preprocessing step to do when
	working with learning algorithms.

	Parameters
	--------------
	X : numpy.ndarray
	    Feature matrix with size (m,n)

	Returns
	--------------
	X_norm : numpy.ndarray
	    Matrix with same shape as that of X containing of 
	    normalized features

	mu : numpy.ndarray
	    vector that contains mean of features

	sigma : numpy.ndarray
	    vector that contains standard deviations of features
	"""

	mu = X.mean(axis=0)
	sigma = X.std(axis=0, ddof=1)
	
	X_norm = (X-mu)/sigma

	return (X_norm, mu, sigma)

def plotFit(min_x, max_x, mu, sigma, theta, p):
	""" Plots a learned polynomial regression fit over an existing figure.
	    Also works with linear regression.

	    Plots the learned polynomial fit with power p and feature 
	    normalization (mu, sigma).

	Parameters
	-------------
	min_x, max_x : float64
	    Minimum and maximum values in X's features
	mu : numpy.ndarray
	    Vector containing of mean values for X's features
	sigma : numpy.ndarray
	    Vector containing of standard deviation for X's features
	theta : parameters of linear regression
	p : int
	     Number of powers

	Return
	--------------
	None
	"""

	theta2D = theta.reshape( theta.shape[0], 1 )
	x = np.arange(min_x - 15, max_x + 25.05, 0.05)
	x = x.reshape( x.size, 1 )

	X_poly = polyFeatures(x, p)
	X_poly = (X_poly - mu)/sigma

	X_poly = np.append( np.ones((x.size,1), dtype=np.float64), 
		                X_poly, axis=1 )
	plt.plot(x, np.dot(X_poly, theta2D) )
	plt.draw()

def validationCurve(X, y, Xval, yval):
	""" Return training errors and validation errors with 
	    different values of lambda
	
	Parameters:
	--------------
	X, y, Xval, yval : numpy.ndarray

	Returns:
	---------------
	lambda_vec, error_train, error_val : numpy.ndarray
	"""

	lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
	error_train = np.zeros( lambda_vec.size )
	error_val   = np.zeros( lambda_vec.size )

	for i in range( lambda_vec.size ):
		theta = trainLinearReg(X, y, lambda_vec[i])
		error_train[i] = linearRegCostFunction(theta, X, y, 0)
		error_val[i] = linearRegCostFunction(theta, Xval, yval, 0)

	return lambda_vec, error_train, error_val

def learningCurveRandom(X, y, Xval, yval, _lambda):
	""" Compute training errors and cross validation errors
	    on randomly selected examples
	"""

	random.seed(9999)
	num_exmp = X.shape[0]
	num_exmp_val = Xval.shape[0]

	error_train = np.zeros(num_exmp)
	error_val   = np.zeros(num_exmp)

	for i in range(num_exmp):
		num_iters = 50
		train_err = 0.0
		val_err   = 0.0
		for t in range(num_iters):
			# randomly select i examples from training and validation set
			train_sel = random.sample(range(0,num_exmp), i+1)
			val_sel = random.sample(range(0,num_exmp_val), i+1)
			theta = trainLinearReg(X[train_sel,:], y[train_sel,:], _lambda)
			train_err += linearRegCostFunction(
				            theta, X[train_sel,:], y[train_sel,:], 0)
			val_err += linearRegCostFunction(
				            theta, Xval[train_sel,:], yval[train_sel,:], 0)
		
		error_train[i] = train_err / num_iters
		error_val[i] = val_err / num_iters

	return (error_train, error_val)


if __name__ == '__main__':

	os.system('cls' if os.name=='nt' else 'clear')

	# =========== Part 1: Loading and Visualizing Data =============
	#   We start the exercise by first loading and visualizing the dataset. 
	#   The following code will load the dataset into your environment and plot
	#   the data.

	print 'Loading and Visualizing Data ...'

	# Load data from ex5data1.mat
	data = loadmat('./ex5data1.mat') 
	X = data['X']
	y = data['y']
	Xval  = data['Xval']
	yval  = data['yval']
	Xtest = data['Xtest']
	ytest = data['ytest']

	m = X.shape[0]

	# Plot training data - Press enter to close the figure 
	# Reference: http://stackoverflow.com/questions/22899723/how-to-close-a-python-figure-by-keyboard-input
	fig = plt.figure()
	plt.scatter(X, y, marker='x', c='r')
	plt.xlabel('Change in water level (x)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.axis([-50, 40, 0, 40])
	plt.draw()
	plt.pause(0.01)
	raw_input('<Press Enter to continue>')
	plt.close(fig)

	# =========== Part 2: Regularized Linear Regression Cost =============
	#   You should now implement the cost function for regularized linear 
	#   regression. 

	theta = np.array( [ 1, 1 ], dtype=np.float64 )
	J = linearRegCostFunction( theta, np.append( np.ones((m,1)), X, axis=1),
	                           y, 1)
	print('\nCost at theta = [1 ; 1]: %f\n'
		  '(this value should be about 303.993192)\n' % J)

	# =========== Part 3: Regularized Linear Regression Gradient =============
	#    You should now implement the gradient for regularized linear 
	#    regression.

	theta = np.array( [ 1, 1 ], dtype=np.float64 )
	grad = linearRegGradient( theta, np.append( np.ones((m,1)), X, axis=1),
	                          y, 1)

	print('\nGradient at theta = [1 ; 1]: [%f %f]\n'
	       '(this value should be about [-15.303016; 598.250744])\n' 
	       % (grad[0], grad[1]));

	# =========== Part 4: Train Linear Regression =============
	#   Once you have implemented the cost and gradient correctly, the
	#   trainLinearReg function will use your cost function to train 
	#   regularized linear regression.
	 
	#   Write Up Note: The data is non-linear, so this will not give a great 
	#                  fit.

	_lambda = 0;
	theta = trainLinearReg(np.append(np.ones((m,1)), X, axis=1), y, _lambda);
	print('Best fit theta = [%f %f]\n' % (theta[0], theta[1]))

	# Plot the fit line
	fig = plt.figure()
	plt.scatter(X, y, marker='x', c='r')
	plt.xlabel('Change in water level (x)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.axis([-50, 40, -5, 40])
	plt.plot(X, np.dot(np.append( np.ones((m,1)), X, axis=1), theta))
	plt.draw()
	plt.pause(0.01)
	raw_input('<Press Enter to continue>')
	plt.close(fig)


	# =========== Part 5: Learning Curve for Linear Regression =============
	#   Next, you should implement the learningCurve function. 

	#   Write Up Note: Since the model is underfitting the data, we expect to
	#                  see a graph with "high bias" -- slide 8 in ML-advice.pdf 

	m = X.shape[0]
	_lambda = 0
	error_train, error_val = learningCurve( 
		                np.append( np.ones( (m,1) ), X, axis=1 ), y,
		                np.append( np.ones((Xval.shape[0],1)), Xval, axis=1 ),
		                yval, _lambda)

	fig = plt.figure()
	plt.plot( range(1,m+1), error_train, range(1,m+1), error_val )
	plt.title('Learning curve for linear regression')
	plt.xlabel('Number of training examples')
	plt.ylabel('Error')
	plt.legend( ['Train', 'Cross Validation'], loc='best' )
	plt.draw()
	plt.pause(0.01)

	print('# Training Examples\tTrain Error\tCross Validation Error')
	for i in range(m):
	    print('  \t%d\t\t%f\t%f' % ( (i+1), error_train[i], error_val[i]) )
	raw_input('<Press Enter to continue>')
	plt.close(fig)

	# =========== Part 6: Feature Mapping for Polynomial Regression =============
	#   One solution to this is to use polynomial regression. You should now
	#   complete polyFeatures to map each example into its powers

	p = 8
	m = X.shape[0]

	# Map X onto Polynomial Features and Normalize
	X_poly = polyFeatures(X, p)
	X_poly, mu, sigma = featureNormalize(X_poly)
	X_poly = np.append( np.ones((m,1), dtype=np.float64), X_poly, axis=1 )

	# Map X_poly_test and normalize (using mu and sigma)
	X_poly_test = polyFeatures(Xtest, p)
	X_poly_test = (X_poly_test - mu)/sigma
	X_poly_test = np.append( 
		np.ones((X_poly_test.shape[0],1), dtype=np.float64), 
		X_poly_test, axis=1 )

	# Map X_poly_val and normalize (using mu and sigma)
	X_poly_val = polyFeatures(Xval, p)
	X_poly_val = (X_poly_val - mu)/sigma
	X_poly_val = np.append( 
		np.ones((X_poly_val.shape[0],1), dtype=np.float64), 
		X_poly_val, axis=1 )

	print('\nNormalized Training Example 1:')
	print X_poly[0,:]
	raw_input('<Press Enter to continue>')


	# =========== Part 7: Learning Curve for Polynomial Regression =============
	#   Now, you will get to experiment with polynomial regression with multiple
	#   values of lambda. The code below runs polynomial regression with 
	#   lambda = 0. You should try running the code with different values of
	#   lambda to see how the fit and learning curve change.

	_lambda = 0
	theta = trainLinearReg(X_poly, y, _lambda)

	print theta

	# Plot training data and fit
	fig = plt.figure()
	plt.scatter(X, y, marker='x', c='r')
	plotFit(X.min(), X.max(), mu, sigma, theta, p)
	plt.xlabel('Change in water level (x)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.title("Polynomial Regression Fit (lambda = %f)" % _lambda)
	# plt.axis([-80, 80, -60, 80])	
	plt.draw()
	plt.pause(0.01)
	raw_input('\n<Press Enter to continue>')
	plt.close(fig)

	# draw learning curve
	m = X.shape[0]
	error_train, error_val = learningCurve( X_poly, y, 
		                             X_poly_val, yval, _lambda)
		                
	fig = plt.figure()
	plt.plot( range(1,m+1), error_train, range(1,m+1), error_val )
	plt.axis([0, 13, 0, 100])
	plt.title('Polynomial Regression Learning Curve (lambda = %f)' 
		       % _lambda)
	plt.xlabel('Number of training examples')
	plt.ylabel('Error')
	plt.legend( ['Train', 'Cross Validation'], loc='best' )
	plt.draw()
	plt.pause(0.01)

	print('# Training Examples\tTrain Error\tCross Validation Error')
	for i in range(m):  
		print('  \t%d\t\t%f\t%f' % ( (i+1), error_train[i], error_val[i]) )
	raw_input('\n<Press Enter to continue>\n')

	# =========== Part 8: Validation for Selecting Lambda =============
	#   You will now implement validationCurve to test various values of 
	#   lambda on a validation set. You will then use this to select the
	#   "best" lambda value.

	lambda_vec, error_train, error_val = validationCurve(X_poly, y, 
		                                                 X_poly_val, yval)

	fig = plt.figure()
	plt.plot( lambda_vec, error_train, lambda_vec, error_val )
	plt.xlabel('lambda')
	plt.ylabel('Error')
	plt.legend(['Train', 'Cross Validation'])
	plt.draw()
	plt.pause(0.01)

	print('# lambda\tTrain Error\tValidation Error');
	for i in range( lambda_vec.size ):
		print('%f\t%f\t%f' % 
			  (lambda_vec[i], error_train[i], error_val[i]) )

	raw_input('\n<Press Enter to continue>\n')

	# ===== Part 9: Optional exercise: Computing test set error ========
	_lambda = 3
	theta = trainLinearReg(X_poly, y, _lambda)
	error_test = linearRegCostFunction(theta, X_poly_test, ytest, 0)

	print 'Test error at (lambda = %f): %f' % (_lambda, error_test)

	raw_input('\n<Press Enter to continue>\n')

	# Part 10: Optional exercise : Plot Learning Curve with
	#          randomly selected examples
	# draw learning curve
	m = X.shape[0]
	_lambda = 0.01
	error_train, error_val = learningCurveRandom( X_poly, y, 
		                             X_poly_val, yval, _lambda)
		                
	fig = plt.figure()
	plt.plot( range(1,m+1), error_train, range(1,m+1), error_val )
	plt.axis([0, 13, 0, 100])
	plt.title('Polynomial Regression Learning Curve (lambda = %f)' 
		       % _lambda)
	plt.xlabel('Number of training examples')
	plt.ylabel('Error')
	plt.legend( ['Train', 'Cross Validation'], loc='best' )
	plt.draw()
	plt.pause(0.01)

	print('# Training Examples\tTrain Error\tCross Validation Error')
	for i in range(m):  
		print('  \t%d\t\t%f\t%f' % ( (i+1), error_train[i], error_val[i]) )
	raw_input('\n<Press Enter to continue>\n')




















