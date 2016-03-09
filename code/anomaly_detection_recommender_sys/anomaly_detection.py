""" Machine Learning Online Class
    Exercise 8 | Anomaly Detection and Collaborative Filtering
    Part 1: Anomaly Detection
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy.io import loadmat
from scipy.stats import multivariate_normal

def estimateGaussian(X):
	""" Esimate data statistics

	Returns
	-------------
	mu, sigma2 : numpy.ndarray
	    Mean and variances of features
	"""

	mu = X.mean( axis=0 )
	sigma2 = X.var( axis=0 )

	return mu, sigma2

def multivariateGaussian(X, mu, sigma2):
	""" Computes the probability density function of the
	    multivariate gaussian distribution.

	    Computes the probability density function of the examples X 
	    under the multivariate gaussian distribution with parameters
	    mu and Sigma2. If Sigma2 is a matrix, it is treated as the 
	    covariance matrix. If Sigma2 is a vector, it is treated
	    as the \sigma^2 values of the variances in each dimension
	    (a diagonal covariance matrix)
	"""

	# Sigma2 = np.array(sigma2)
	# k = len(mu)
	# if ( np.ndim(Sigma2) == 1 or Sigma2.shape[0] == 1 
	#  	or Sigma2.shape[1] == 1 ):
	#  	Sigma2 = np.diag(Sigma2)


	# Xn = X - mu
	# p = ( (2 * np.pi) ** (- k / 2) * linalg.det(Sigma2) ** (-0.5) 
	#   	 * np.exp(-0.5 * np.sum(np.dot(Xn, linalg.pinv(Sigma2) ) 
	#                               * Xn, axis=1)) )

	# p can be calculated very quickly using the following function
	# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
	p = multivariate_normal.pdf(X, mean=mu, cov=sigma2)
	return p

def selectThreshold(yval, pval):
	""" Find the best threshold to use for detecting outliers
	    The best threshold is chosen based on cross validation data
	
	Parameters
	-------------
	yval : numpy.ndarray
		Contain values 1 (for anomalous examples) and 0 (normal cases)
	pval : numpy.ndarray
	    Probabilities of examples in cross validation data

	Returns
	-------------
	epsilon : numpy.float64
	    Best threshold
	f1 : numpy.float64
	    F1 score on cross validation data
	"""

	bestEpsilon = 0.0
	bestF1 = 0.0
	stepsize = ( np.max(pval) - np.min(pval) ) / 1000

	for epsilon in np.arange(np.min(pval), np.max(pval) + stepsize, stepsize, dtype=np.float64):
		predictions = pval < epsilon

		tp = np.logical_and( (predictions == True),  (yval == 1) ).sum()
		fp = np.logical_and( (predictions == True),  (yval == 0) ).sum()
		fn = np.logical_and( (predictions == False), (yval == 1) ).sum()
		prec = (0.0 if tp+fp == 0 else 1.0*tp/(tp+fp))
		rec  = (0.0 if tp+fn == 0 else 1.0*tp/(tp+fn))
		F1   = (0.0 if prec+rec == 0 else 2.0*prec*rec/(prec+rec))
		if F1 > bestF1:
			bestF1 = F1
			bestEpsilon = epsilon

	return bestEpsilon, bestF1

def visualizeFit(X,  mu, sigma2):
	""" Visualize the dataset and its estimated distribution.
	    VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you 
	    the probability density function of the Gaussian distribution.
	    Each example has a location (x1, x2) that depends on 
	    its feature values.
	"""
	X1, X2 = np.mgrid[0:35.5:.5,0:35.5:.5]
	Z = multivariateGaussian(np.r_['0,2', X1.flatten(), X2.flatten()].T,
		                     mu, sigma2)

	Z = Z.reshape( X1.shape )
	Z = Z.T
	
	plt.plot(X[:, 0], X[:, 1],'bx')
	plt.hold(True)
	plt.contour(X1, X2, Z, 10 ** np.arange(-20,0,3, dtype=np.float64) )
	plt.draw()
	plt.pause(0.01)
	# if (sum(isinf(Z)) == 0)
	#     contour(X1, X2, Z, 10.^(-20:3:0)');

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')

	# ============= Part 1: Load Example Dataset  ================
	#  We start this exercise by using a small dataset that is easy to
	#  visualize.
	#  Our example case consists of 2 network server statistics across
	#  several machines: the latency and throughput of each machine.
	#  This exercise will help us find possibly faulty (or very fast) machines.

	print('Visualizing example dataset for outlier detection.\n')

	data1 = loadmat('ex8data1.mat')
	X = data1['X']
	Xval = data1['Xval']
	yval = data1['yval']
	yval = yval.flatten()
	print 'Print shape of X'
	print X.shape; print
	
	# Visualize the example dataset
	fig = plt.figure(1)
	plt.plot(X[:, 0], X[:, 1], 'bx')
	plt.axis([0, 30, 0, 30])
	plt.xlabel('Latency (ms)')
	plt.ylabel('Throughput (mb/s)')
	plt.draw()
	plt.pause(0.01)
	raw_input('Program paused. Press enter to continue.\n')
	plt.close(fig)

	# ====== Part 2: Estimate the dataset statistics ======
	# For this exercise, we assume a Gaussian distribution for the dataset.
	# We first estimate the parameters of our assumed Gaussian distribution, then compute the probabilities for each of the points and then visualize both the overall distribution and where each of the points falls in terms of that distribution.

	print('Visualizing Gaussian fit.\n')

	# Estimate mu and sigma2
	mu, sigma2 = estimateGaussian(X)

	# Returns the density of the multivariate normal at each 
	# data point (row) of X
	p = multivariateGaussian(X, mu, sigma2)

	# Visualize the fit
	plt.hold(True)
	fig = plt.figure(2)
	visualizeFit(X,  mu, sigma2)
	plt.xlabel('Latency (ms)')
	plt.ylabel('Throughput (mb/s)')
	raw_input('Program paused. Press enter to continue.\n')

	# ================== Part 3: Find Outliers ===============
	# Now you will find a good epsilon threshold using a cross-validation
	# set probabilities given the estimated Gaussian distribution

	pval = multivariateGaussian(Xval, mu, sigma2)

	epsilon, F1 = selectThreshold(yval, pval)

	print('Best epsilon found using cross-validation: %e' % epsilon)
	print('Best F1 on Cross Validation Set:  %f' % F1)
	print('   (you should see a value epsilon of about 8.99e-05)\n')
	
	# Find the outliers in the training set and plot the
	outliers = (p < epsilon).nonzero()

	# Draw a red circle around those outliers
	plt.plot(X[outliers, 0], X[outliers, 1], 'r^', linewidth=2,
		     markersize=10, markerfacecolor='None', markeredgecolor='r')
	plt.draw()
	plt.pause(0.01)
	raw_input('Program paused. Press enter to continue.\n')

	# ======= Part 4: Multidimensional Outliers ===========
	# We will now use the code from the previous part and apply it to a 
	# harder problem in which more features describe each datapoint 
	# and only some features indicate whether a point is an outlier.

	data2 = loadmat('ex8data2.mat');
	X = data2['X']
	Xval = data2['Xval']
	yval = data2['yval']
	yval = yval.flatten()

	print 'Shape of X: ' 
	print X.shape; print

	# Apply the same steps to the larger dataset
	mu, sigma2 = estimateGaussian(X)

	# Training set 
	p = multivariateGaussian(X, mu, sigma2)
	# Cross-validation set
	pval = multivariateGaussian(Xval, mu, sigma2)

	# Find the best threshold
	epsilon, F1 = selectThreshold(yval, pval)

	print('Best epsilon found using cross-validation: %e' % epsilon);
	print('Best F1 on Cross Validation Set:  %f' % F1)
	print('# Outliers found: %d' % np.sum(p < epsilon))
	print('   (you should see a value epsilon of about 1.38e-18)\n')






