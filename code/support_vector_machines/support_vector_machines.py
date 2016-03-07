
"""
   Machine Learning Online Class
   Exercise 6 | Support Vector Machines

   In this python script, we use scikit-learn library for training SVM.
   The next task in our to-do-list is to implement SMO algorithm for
   training SVM.
"""

import os
import sys
import matplotlib.pyplot as plt
from sklearn import svm
import itertools
import numpy as np
from scipy.io import loadmat

def plot_data(X, y):
    """ Plot the data points X and y into a new figure 
        Plot the data points with + for the positive examples
        and o for the negative examples. X is assumed to be a Mx2 matrix.
    """

    pos = y == 1
    neg = y == 0
    plt.scatter( X[pos,0], X[pos,1], marker='+', c='b')
    plt.scatter( X[neg,0], X[neg,1], c='y')
    return plt

def visualizeBoundaryLinearSklearn(X, y, clf):
	""" Visualize linear decision boundary

	Parameters:
	-------------
	X, y : numpy.ndarray
	       Training data and labels
	clf  : svm.SVC object
	       Learned model from the data

	Returns
	-------------
	None
	"""

	w = clf.coef_.flatten()
	b = clf.intercept_
	xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
	yp = -(b + w[0] * xp)/w[1]
	plot_data(X, y)
	plt.plot(xp, yp)

def visualizeBoundarySklearn(X, y, clf):
	""" Visualize non-linear decision boundary

	Parameters:
	--------------
	X, y : numpy.ndarray
	       Training data and labels
	clf  : svm.SVC object
	       Learned model from the data

	Returns
	-------------
	None
	"""

	print 'Visualizing non-linear decision boundary ...'
	plot_data(X, y)
	x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
	x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
	xx1, xx2 = np.meshgrid(x1plot, x2plot)

	z = np.zeros( shape=(len(x1plot), len(x2plot)) )
	for i in xrange(len(x1plot)):
		for j in xrange(len(x2plot)):
			x = np.array( [x1plot[i], x2plot[i]] ).reshape(1,2)
			yp = clf.predict(x)
			z[i,j] = yp[0]
	z = z.T	
	plt.contour(x1plot, x2plot, z) 

def gaussianKernel(x1, x2, sigma):
	""" Compute Gaussian kernel between two examples x1, x2

	Parameters
	-----------
	x1, x2 : numpy.ndarray
	    Two column vector (2D)
	sigma : float64
	    Parameter in Gaussian kernel function (RBF)

	Returns
	-----------
	Gaussian kernel value
	"""

	return np.exp( - np.sum( (x1 - x2) ** 2 )/(2 * (sigma ** 2)) );

if __name__ == '__main__':
	os.system('cls' if os.name=='nt' else 'clear')

	# ========= Part 1: Loading and Visualizing Data ============
	# We start the exercise by first loading and visualizing the dataset. 
	# The following code will load the dataset into your environment and plot the data.

	print 'Loading and Visualizing Data ...'
	data = loadmat('ex6data1.mat')
	X = data['X']
	y = data['y']
	# in python numpy, it is preferred to use 1D array
	y = y.flatten()

	print 'Print shape of (X, y)'
	print X.shape
	print y.shape

	print '\nPlot training data'

	fig = plt.figure()
	plot_data(X, y)
	plt.axis([0, 4.5, 1.5, 5])
	plt.draw()
	plt.pause(0.05)
	raw_input('<Press Enter to continue>')
	plt.close(fig)

	# ================= Part 2: Training Linear SVM ===============
	#  The following code will train a linear SVM on the 
	#  dataset and plot the decision boundary learned.

	print '\nTraining Linear SVM'

	for C in [1.0, 1000]:
		print 'C = %f' % C
		clf = svm.SVC(C, kernel='linear')
		clf.fit(X, y)
		fig = plt.figure()
		visualizeBoundaryLinearSklearn(X, y, clf)
		plt.axis([0, 4.5, 1.5, 5])
		plt.draw()
		plt.pause(0.05)
		raw_input('<Press Enter to continue>')
		plt.close(fig)


	# =========== Part 3: Implementing Gaussian Kernel ===============
	#  You will now implement the Gaussian kernel to use
	#  with the SVM. You should complete the code in gaussianKernel.m
	print('\nEvaluating the Gaussian Kernel ...')
	x1 = np.array([1, 2, 1], dtype=np.float64).reshape(3, 1)
	x2 = np.array([0, 4, -1], dtype=np.float64).reshape(3, 1)
	sigma = 2.0;
	sim = gaussianKernel(x1, x2, sigma);

	print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], '
	      'sigma = 2 :\n\t%f\n'
	      '(this value should be about 0.324652)' % sim);

	## =============== Part 4: Visualizing Dataset 2 ================
	#  The following code will load the next dataset into your environment and plot the data. 

	print '\nLoading and Visualizing Data ...'
	data = loadmat('ex6data2.mat')
	X = data['X']
	y = data['y']
	y = y.flatten()
	print 'Print shape of (X, y)'
	print X.shape
	print y.shape

	fig = plt.figure()
	plot_data(X, y)
	plt.axis([0, 1, 0.4, 1])
	plt.draw()
	plt.pause(0.05)
	raw_input('<Press Enter to continue>')
	plt.close(fig)

	# ===== Part 5: Training SVM with RBF Kernel (Dataset 2) =====
	#  After you have implemented the kernel, we can now 
	#  use it to train the SVM classifier.

	# In this version, use scikit-learn library, next we will try to
	# use  custom kernel and compare with the result produced by
	# the standard library

	# Ignore contour plot to move on other parts
	# http://stackoverflow.com/questions/30029012/plotting-curve-decision-boundary-in-python-using-matplotlib
	print('\nTraining SVM with RBF Kernel ...');
	C = 1.0
	sigma = 0.1
	clf = svm.SVC(C, kernel='rbf', gamma=sigma)
	clf.fit(X, y)
	fig = plt.figure()
	visualizeBoundarySklearn(X, y, clf)
	plt.axis([0, 1, 0.4, 1])
	plt.draw()
	plt.pause(0.05)
	raw_input('<Press Enter to continue>')


	# =============== Part 6: Visualizing Dataset 3 ================
	# The following code will load the next dataset into your environment and plot the data.

	print '\nLoading and Visualizing Data ...'
	data = loadmat('ex6data3.mat')
	X = data['X']
	y = data['y']
	Xval = data['Xval']
	yval = data['yval']
	y = y.flatten()
	yval = yval.flatten()
	print 'Print shape of (X, y)'
	print X.shape
	print y.shape

	print 'Print shape of (Xval, yval)'
	print Xval.shape
	print yval.shape

	fig = plt.figure()
	plot_data(X, y)
	plt.draw()
	plt.pause(0.05)
	raw_input('<Press Enter to continue>')
	plt.close(fig)

 	# ===== Part 7: Training SVM with RBF Kernel (Dataset 3) ======
 	#  This is a different dataset that you can use to experiment with. #  Try  different values of C and sigma here.

 	params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

 	C = None
 	g = None
 	maxscore = -1
 	for c, g in itertools.product(params, params):
 		clf = svm.SVC(c, kernel='rbf', gamma=g)
 		clf.fit(X, y)
 		acc = clf.score(Xval, yval)
 		# print (c, g, acc)
 		if acc > maxscore:
 			C = c
 			g = g
 			maxscore = acc

 	print('Best params on validation C = %f, gamma = %f\n'
 	      'Accuracy on the validation set: %f' % (C, g, maxscore)) 

 	clf = svm.SVC(C, kernel='rbf', gamma=g)
 	clf.fit(X, y)
 	fig = plt.figure()
	visualizeBoundarySklearn(X, y, clf)
	plt.draw()
	plt.pause(0.05)
	raw_input('<Press Enter to continue>')

























