"""
Python code for the first part of programming exercise 3 
Multi-class classification and Neural Networks
(Machine Learning course)
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_bfgs, fmin

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) ) 

def cost_function(theta, X, y, _lambda):
    m = y.shape[0]
    J = 0
    hx = sigmoid( np.dot(X, theta) ) # hx.shape m x 1
    y_flat = y.flatten()
    J = np.sum( -y_flat * np.log(hx) - (1-y_flat) * np.log(1-hx) )/m
    J += _lambda * (theta[1:] ** 2).sum()/(2*m)
    return J

# theta.shape (n,) (ndim=1)
# X.shape = m x n (ndim = 2)
# y.shape = m x 1 (ndim =2)
# return gradient vector with shape (n,) (ndim=1)
def gradient(theta, X, y, _lambda):
    m = y.shape[0]
    n = theta.shape[0]
    grad = np.zeros( n )
    hx = sigmoid( np.dot(X, theta) ) # hx.shape (m, ) ndim = 1
    hx = hx.reshape( (m, 1) )
    grad = np.dot(X.T, hx - y) / m
    
    grad = grad.flatten()
    temp = np.array(theta)
    temp[0] = 0
    grad += _lambda * temp / m
    
    return grad

def oneVsAll(X, y, num_labels, _lambda):
    m = X.shape[0] # number of training examples
    n = X.shape[1] # number of features
    all_theta = np.zeros( (num_labels, n+1), dtype=np.float64 )
    # add ones to X data matrix
    X_new = np.concatenate( (np.ones( (m,1) ), X), axis=1 )
    
    for c in xrange(1, num_labels+1):
    	print 'Training classifier for class %d ...' % c
        initial_theta = np.zeros( n+1, dtype=np.float64 )
        y_lb = np.array( (y == c), dtype=np.int )
        theta = fmin_bfgs(cost_function, initial_theta,
                          fprime=gradient, args=(X_new, y_lb, _lambda),
                          disp=False)
        all_theta[c-1] = theta.reshape( (1, n+1) )
    
    return all_theta

# all_theta.shape = num_labels x (n+1)
# X has shape: m x n
# return prediction with shape m x 1
def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros( (m, 1), dtype=np.float32 )
    
    X_new = np.concatenate( (np.ones( (m,1) ), X), axis=1 )
    prob = sigmoid( np.dot(X_new, all_theta.T) )
    p = np.argmax( prob, axis=1 ) + 1
    return p.reshape( (m, 1) )

if __name__ == '__main__':
	mat_dict = loadmat('ex3data1.mat')
	X = mat_dict['X']
	y = mat_dict['y']
	m = X.shape[0]

	num_labels = 10
	_lambda = 0.1

	print 'Training One-vs-All Logistic Regression...'
	all_theta = oneVsAll(X, y, num_labels, _lambda)

	pred = predictOneVsAll(all_theta, X)
	print('Training accuracy (with fmin_bfgs): %s' % 
	      ( 100 * (pred == y).mean() ) )



