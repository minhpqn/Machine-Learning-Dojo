""" Machine Learning Online Class
    Exercise 7 | Principle Component Analysis and K-Means Clustering
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.cm as cm
import math

def featureNormalize(X):
	""" Perform feature normalization
	    Subtract the mean and divide by the standard deviation

	Returns
	-----------
	X_norm : numpy.ndarray
	mu, sigma
	"""

	mu = X.mean(axis = 0)
	sigma = X.std(axis = 0, ddof=1)
	X_norm = (X - mu)/sigma

	return X_norm, mu, sigma

def pca(X):
	""" PCA Run principal component analysis on the dataset X
	    [U, S, X] = pca(X) computes eigenvectors of 
	                       the covariance matrix of X
	Returns
	--------------
	U, S: numpy.ndarray

	the eigenvectors U
	the eigenvalues (on diagonal) in S
	"""

	m, n = X.shape

	# Instructions: You should first compute the covariance matrix. 
	# Then, you should use the "svd" function to compute the 
	# eigenvectors and eigenvalues of the covariance matrix. 
	# Note: When computing the covariance matrix, remember to 
	#       divide by m (the number of examples).

	from scipy import linalg

	Sigma = np.dot(X.T, X)/m
	U, S, V = linalg.svd(Sigma)

	return U, S

def drawLine(p1, p2, linestyle='k-'):
	plt.plot([ p1[0], p2[0] ], [ p1[1], p2[1] ], linestyle)

def projectData(X_norm, U, K):
	""" Project the data onto the principal components

	Parameters
	-------------
	X_norm : numpy.ndarray
	   normalized feature data
	U : numpy.ndarray
	   principal components
	K : number of desired components

	Return
	-------------
	Z : numpy.ndarray
	   Projected data of X_norm onto reduced of U 
	   (contains K components)
	Ureduce = U[:,0:K]
	Z = Ureduce' * X
	"""

	return np.dot(X_norm, U[:,0:K])

def recoverData(Z, U):
	""" Recover data by projecting them back to the original 
	    high dimensional space

	Return:
	-----------
	X_rec : numpy.ndarray
	"""
	K = Z.shape[1]
	return np.dot(Z, U[:,0:K].T)

def displayData(X, show=True):
	""" Visualize Face Image dataset
	"""

	example_width = int( round(math.sqrt(X.shape[1])) )

	import matplotlib.cm as cm

	# Compute rows, cols
	m, n = X.shape
	example_height = (n / example_width)

	# Compute number of items to display
	display_rows = int( math.floor(math.sqrt(m)) )
	display_cols = int( math.ceil(1.0 * m / display_rows) )

	# Between images padding
	pad = 1

	# Setup blank display
	nrows = pad + display_rows * (example_height + pad)
	ncols = pad + display_cols * (example_width + pad)
	display_array = - np.ones( (nrows, ncols) )

	# Copy each example into a patch on the display array
	curr_ex = 0
	for i in xrange(display_rows):
		for j in xrange(display_cols):
			if curr_ex >= m:
				break
			# Copy the patch
			# Get the max value of the patch
			max_val = np.max( np.abs( X[curr_ex,:] ) )
			r_start = pad + i * (example_height + pad)
			r_end   = (i+1) * (example_height + pad)
			c_start = pad + j * (example_width + pad)
			c_end   = (j+1) * (example_width + pad)

			display_array[r_start:r_end, c_start:c_end]= ( 
				  X[curr_ex, :].reshape(
				     (example_width, example_height) ).T / max_val )
			
			curr_ex = curr_ex + 1

		if curr_ex >= m:
			break

	if show:
		# Display Image
		plt.imshow(display_array, cm.Greys_r)
		plt.axis('off')

	return display_array
	
if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')

	# ================== Part 1: Load Example Dataset  =============
	# We start this exercise by using a small dataset that is easily to
	# visualize

	print('Visualizing example dataset for PCA.\n')

	data = loadmat('ex7data1.mat')
	X = data['X']
	# Visualize the example dataset
	fig = plt.figure(1)
	plt.axis([0.5, 6.5, 2, 8])
	plt.plot(X[:, 0], X[:, 1], 'bo')
	plt.draw()
	plt.pause(0.01)
	raw_input('<Program paused. Press enter to continue>\n')
	plt.close(fig)

	# ============= Part 2: Principal Component Analysis =============
	#  You should now implement PCA, a dimension reduction technique. 
	#  You should complete the code in pca.m

	print('\nRunning PCA on example dataset.\n');

	# Before running PCA, it is important to first normalize X
	X_norm, mu, sigma = featureNormalize(X)

	# Run PCA
	U, S = pca(X_norm)

	# Compute mu, the mean of the each feature
	# Draw the eigenvectors centered at mean of data. 
	# These lines show the directions of maximum 
	# variations in the dataset.

	fig = plt.figure(2)
	plt.axis([0.5, 6.5, 2, 8])
	plt.plot(X[:, 0], X[:, 1], 'bo')
	drawLine(mu, mu + 1.5 * S[0] * U[:,0])
	drawLine(mu, mu + 1.5 * S[1] * U[:,1])
	plt.draw()
	plt.pause(0.01)

	print('Top eigenvector: ')
	print(' U(:,1) = %.6f %.6f' % (U[0,0], U[1,0]))
	print(' (you should expect to see -0.707107 -0.707107)\n')
	raw_input('<Program paused. Press enter to continue>\n')
	plt.close(fig)
	
	# ============= Part 3: Dimension Reduction ==================
	# You should now implement the projection step to map the data onto the first k eigenvectors. The code will then plot the data in this reduced dimensional space.  This will show you what the data looks like when using only the corresponding eigenvectors to reconstruct it.

	print('\nDimension reduction on example dataset.\n');

	# Plot the normalized dataset (returned from pca)
	fig = plt.figure()
	plt.plot(X_norm[:, 0], X_norm[:, 1], 'b^');
	plt.axis([-4, 3, -4, 3])
	plt.draw()
	plt.pause(0.01)

	plt.hold(True)

	# Project the data onto K = 1 dimension
	K = 1
	Z = projectData(X_norm, U, K)
	print('Projection of the first example: %f' % Z[0])
	print('(this value should be about 1.481274)\n')

	X_rec  = recoverData(Z, U)
	print( 'Approximation of the first example: %f %f' %
		   (X_rec[0, 0], X_rec[0, 1]) )
	print('(this value should be about  -1.047419 -1.047419)\n')

	# Draw lines connecting the projected points to the original points

	plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
	plt.draw()
	plt.pause(0.01)
	
	for i in range(X_norm.shape[0]):
		drawLine(X_norm[i,:], X_rec[i,:], 'k--')

	plt.draw()
	plt.hold(False)
	raw_input('<Program paused. Press enter to continue>\n')
	plt.close(fig)

	# ====== Part 4: Loading and Visualizing Face Data =======
	# We start the exercise by first loading and 
	# visualizing the dataset.
	
	print('\nLoading face dataset.\n')

	# Load Face dataset
	facedata = loadmat('ex7faces.mat')
	X = facedata['X']
	print 'Print shape of face data'
	print X.shape

	# Display the first 100 faces in the dataset
	fig = plt.figure()
	displayData( X[0:100,:] )
	plt.draw()
	plt.pause(0.01)
	raw_input('<Program paused. Press enter to continue>\n')
	plt.close(fig)

	# ====== Part 5: PCA on Face Data: Eigenfaces  =======
	# Run PCA and visualize the eigenvectors which are in this case eigenfaces
	# We display the first 36 eigenfaces.
	
	print('\nRunning PCA on face dataset...')
	
	# Before running PCA, it is important to first normalize X by subtracting the mean value from each feature
	X_norm, mu, sigma = featureNormalize(X)
	# Run PCA
	U, S = pca(X_norm)

	# Visualize the top 36 eigenvectors found
	fig = plt.figure()
	displayData(U[:, 0:36].T)
	plt.draw()
	plt.pause(0.01)
	raw_input('<Program paused. Press enter to continue>\n')
	plt.close(fig)

	# ====== Part 6: Dimension Reduction for Faces =============
	# Project images to the eigen space using the top k eigenvectors 
	# If you are applying a machine learning algorithm 

	print('\nDimension reduction for face dataset.\n')
	K = 100
	Z = projectData(X_norm, U, K)

	sys.stdout.write('The projected data Z has a size of: ')
	print Z.shape
	raw_input('\n<Program paused. Press enter to continue>\n')

	# Part 7: Visualization of Faces after PCA Dimension Reduction
	# Project images to the eigen space using the top 
	# K eigen vectors and visualize only using those K dimensions 
	# Compare to the original input, which is also displayed

	print('\nVisualizing the projected (reduced dimension) faces.\n')

	K = 100
	X_rec  = recoverData(Z, U)
	
	# Display normalized data
	fig = plt.figure()
	plt.subplot(1, 2, 1)
	orig_img = displayData(X_norm[0:100,:], False)	
	plt.imshow(orig_img, cm.Greys_r)
	plt.axis('off')
	plt.draw()
	plt.title('Original faces')

	# Display reconstructed data from only k eigenfaces
	plt.subplot(1, 2, 2)
	rec_img = displayData(X_rec[0:100,:], False)
	plt.imshow(rec_img, cm.Greys_r)
	plt.axis('off')
	plt.draw()
	plt.title('Recovered faces')
	plt.pause(0.01)
	raw_input('<Program paused. Press enter to continue>\n')	
	plt.close(fig)

	# Part 8(a): Optional (ungraded) Exercise: PCA for Visualization
	# One useful application of PCA is to use it to visualize 
	# high-dimensional data. In the last K-Means exercise you ran 
	# K-Means on 3-dimensional pixel colors of an image. 
	# We first visualize this output in 3D, and then 
	# apply PCA to obtain a visualization in 2D.

	os.system('cls' if os.name == 'nt' else 'clear')

	# Re-load the image from the previous exercise and 
	# run K-Means on it for this to work, you need to 
	# complete the K-Means assignment first

	from kmeans import kMeansInitCentroids, runkMeans, plot_data_points
	import scipy.misc	
	A = scipy.misc.imread('bird_small.png')
	A = A / 255.0
	img_size = A.shape
	X = A.reshape( img_size[0] * img_size[1], 3 )
	K = 16
	max_iters = 10
	initial_centroids = kMeansInitCentroids(X, K)
	centroids, idx = runkMeans(X, initial_centroids, max_iters)

	# Sample 1000 random indexes (since working with all the data is
	# too expensive. If you have a fast computer, you may increase this.

	# Draw 3D using mplot3d
	from mpl_toolkits.mplot3d import Axes3D

	sel = np.random.choice( range(X.shape[0]), size=1000, 
		                    replace=False )

	# http://matplotlib.org/examples/mplot3d/scatter3d_demo.html
	fig = plt.figure()
	ax  = fig.add_subplot(111, projection='3d')

	# Visualize the data and centroid memberships in 3D
	ax.scatter( X[sel, 0], X[sel, 1], X[sel, 2], c=idx[sel] )
	plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
	plt.draw()
	plt.pause(0.01)
	raw_input('<Program paused. Press enter to continue>\n')
	plt.close(fig)


	# Part 8(b): Optional (ungraded) Exercise: PCA for Visualization
	# Use PCA to project this cloud to 2D for visualization

	# Subtract the mean to use PCA
	X_norm, mu, sigma = featureNormalize(X)

	# PCA and project the data to 2D
	U, S = pca(X_norm)
	Z = projectData(X_norm, U, 2)

	# Plot in 2D
	fig = plt.figure()
	plot_data_points(Z[sel, :], idx[sel], K)
	plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
	plt.draw()
	plt.pause(0.01)
	raw_input('<Program paused. Press enter to continue>\n')


