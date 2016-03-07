""" Machine Learning Online Class
    Exercise 7 | Principle Component Analysis and K-Means Clustering
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.lines import Line2D


def findClosestCentroids(X, centroids):
	""" Find closet centroids for examples in X
	Parameters
	------------
	X : numpy.ndarray
	    Data matrix
	centroids : numpy.ndarray
	    centroids in k-means algorithm

	Return
	------------
	idx : numpy.ndarray (K elements)
	    centroid assignment for examples in X
	    idx[i] is the centroid of example i-th in X
	"""
	k = centroids.shape[0]
	idx = np.zeros(X.shape[0], dtype=np.int)
	for i in xrange(X.shape[0]):
		dist = np.sum( (X[i] - centroids) ** 2, axis=1 )
		idx[i] = np.argmin(dist)

	return idx

def computeCentroids(X, idx, K):
	""" Compute centroids from current assignments of examples

	Parameters
	--------------
	X : numpy.ndarray
	    Data matrix
	idx : numpy.ndarray
	    centroid assignment for examples in X
	    idx[i] is the centroid of example i-th in X
	K : int
	    The number of centroids

	Return
	--------------
	centroids : numpy.ndarray
	    new centroids (2D array) (shape: (K,2))	    
	"""

	centroids = np.zeros( (K, X.shape[1]) )
	for k in xrange(K):
		Ck = [i for i in xrange(idx.size) if idx[i] == k]
		centroids[k,:] = np.mean( X[Ck,:], axis = 0 )

	return centroids

def plot_data_points(X, idx, k):
	""" Plot data points, coloring them so that points with the same
	    centroid assignments have the same color

	Parameters
	------------
	idx : numpy.ndarray
	      Centroid assignment, idx[i] is the centroid of 
	      the example i-th in X
	k : int
	      The number of centroids
	"""

	plt.scatter(X[:,0], X[:,1], c=idx, marker='^', s=30)

def plotProgresskMeans(X, centroids, previous_centroids, idx, k, i):
	""" Plot the progress when running k-means

	Parameters
	---------------
	X : numpy.ndarray
	centroid, previous_centroids : numpy.ndarray
	idx : numpy.ndarray
	      centroid assignment of examples
	k : int
	    number of centroids
	i : iteration number

	Return
	---------------
	None
	"""

	plot_data_points(X, idx, k)

	plt.scatter(centroids[:,0], centroids[:,1], marker='x', 
		        s = 70, c='k')

	for j in xrange( centroids.shape[0] ):
		plt.plot([ centroids[j,0], previous_centroids[j,0] ],
		         [ centroids[j,1], previous_centroids[j,1] ], c='b')

	plt.title('Iteration %d' % (i+1))
	plt.draw()
	plt.pause(0.01)

def runkMeans(X, initial_centroids, max_iters, plotProgress=False):
	""" Run K-means algorithms

	Parameters
	--------------
	X : numpy.ndarray
	    Data matrix
	initial_centroids : numpy.ndarray
	max_iters : int
	plotProgress : (True or False)
	    Indicate whether we plot the progress of K-means algorithm

	Returns
	---------------
	centroids : numpy.ndarray
	    Final centroids learned from the data
	idx : numpy.ndarray
		Centroid assignment for examples in X
	    idx[i] is the centroid of example i-th in X
	"""

	centroids = initial_centroids
	previous_centroids = centroids
	k = centroids.shape[0]
	prev_idx = np.zeros(X.shape[0])
	for i in xrange(max_iters):
		# Output progress
		print('K-means iteration %d/%d...' % ( (i+1), max_iters) )

		idx = findClosestCentroids(X, centroids)

		if ( np.array_equal(idx, prev_idx) ):
			break
		else:
			prev_idx = idx

		if plotProgress:
			plotProgresskMeans(X, centroids, previous_centroids, 
				               idx, k, i)
			previous_centroids = centroids
			raw_input('Press enter to continue.')

		centroids = computeCentroids(X, idx, k)

	return centroids, idx

def kMeansInitCentroids(X, K):
	""" Random initialization for k-means algorithms
	    Just return a randomly sample of k examples in X
	"""
	np.random.seed(99999)
	randidx = np.random.permutation( range(X.shape[0]) )

	return X[randidx[0:K], :]

def kMeansOnPixels(filename, K, max_iters):
	""" Apply K-means algorithm for compressing images
	"""

	# ============= Part 4: K-Means Clustering on Pixels =============
	# In this exercise, you will use K-Means to compress an image. 
	# To do this, you will first run K-Means on the colors of 
	# the pixels in the image and then you will map each pixel on 
	# to it's closest centroid.

	print('\nRunning K-Means clustering on pixels from an image.')
	print('Image file: %s, K = %d, max_iters=%d\n' % 
		 (filename, K, max_iters))

	# Load an image of a bird
	import scipy.misc	
	A = scipy.misc.imread(filename)
	img_size = A.shape
	A = A / 255.0

	# Reshape the image into an Nx3 matrix where N = number of pixels.
	# Each row will contain the Red, Green and Blue pixel values
	# This gives us our dataset matrix X that we will use K-Means on.
	X = A.reshape( A.shape[0] * A.shape[1], 3 )

	# When using K-Means, it is important the initialize the centroids
	# randomly. 
	# You should complete the code in kMeansInitCentroids.m before proceeding

	initial_centroids = kMeansInitCentroids(X, K)

	# Run K-Means
	centroids, idx = runkMeans(X, initial_centroids, max_iters)

	raw_input('<Program paused. Press enter to continue>')


	# ========== Part 5: Image Compression ===================
	# In this part of the exercise, you will use the clusters of 
	# K-Means to compress an image. To do this, we first find the
	# closest clusters for each example

	print('Applying K-Means to compress an image.\n')

	# Find closest cluster members
	idx = findClosestCentroids(X, centroids)

	# Essentially, now we have represented the image X as in terms of the indices in idx. 
	# We can now recover the image from the indices (idx) by mapping each pixel (specified by it's index in idx) to the centroid value

	X_recovered = centroids[idx,:]

	# Reshape the recovered image into proper dimensions
	X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

	# Display the original image 
	# Use subplot to plot two images
	# Reference: http://bicycle1885.hatenablog.com/entry/2014/02/14/023734
	plt.subplot(1, 2, 1)
	plt.imshow(A)
	plt.draw()
	plt.pause(0.01)
	plt.title('Original image')

	# Display compressed image side by side
	plt.subplot(1, 2, 2)
	plt.imshow(X_recovered)
	plt.draw()
	plt.title('Compressed, with %d colors.' % K)

	raw_input('<Program paused. Press enter to continue>\n')


if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	
	# ================= Part 1: Find Closest Centroids =================
	#   To help you implement K-Means, we have divided the learning algorithm 
	#   into two functions -- findClosestCentroids and computeCentroids. In this part, you shoudl complete the code in the findClosestCentroids function. 

	print('Finding closest centroids.\n')
	data = loadmat('ex7data2.mat')
	X = data['X']

	print 'Print shape of data X'
	print X.shape

	K = 3 # 3 Centroids
	initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

	# Find the closest centroids for the examples using the initial_centroids

	idx = findClosestCentroids(X, initial_centroids)
	print('Closest centroids for the first 3 examples: ')
	print idx[0:3] + 1
	print('(the closest centroids should be 1, 3, 2 respectively)')
	raw_input('<Program paused. Press enter to continue>\n')

	# ================= Part 2: Compute Means ======================
	#  After implementing the closest centroids function, you should now
	#  complete the computeCentroids function.

	print('\nComputing centroids means.\n')

	#Compute means based on the closest centroids found in the previous part.

	centroids = computeCentroids(X, idx, K)

	print('Centroids computed after initial finding of closest centroids:')
	print centroids
	print('\n(the centroids should be')
	print('   [ 2.428301 3.157924 ]')
	print('   [ 5.813503 2.633656 ]')
	print('   [ 7.119387 3.616684 ]\n')

	# ========== Part 3: K-Means Clustering ============
	# After you have completed the two functions computeCentroids and
	# findClosestCentroids, you have all the necessary pieces to run the
	# kMeans algorithm. In this part, you will run the K-Means algorithm on the example dataset we have provided. 

	print('\nRunning K-Means clustering on example dataset.\n')

	# Load an example dataset
	data = loadmat('ex7data2.mat')
	X = data['X']

	# Settings for running K-Means
	K = 3
	max_iters = 10

	# For consistency, here we set centroids to specific values
	# but in practice you want to generate them automatically, such as by settings them to be random examples (as can be seen in kMeansInitCentroids).
	
	initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
	
	# initial_centroids = kMeansInitCentroids(X, K)

	# Run K-Means algorithm. The 'true' at the end tells our function to plot the progress of K-Means
	fig = plt.figure()
	centroids, idx = runkMeans(X, initial_centroids, max_iters, True)

	print('\nK-Means Done.\n')
	raw_input('<Program paused. Press enter to continue>\n')
	plt.close(fig)

	filenames = [ 'bird_small.png', 'kyoto.jpg' ]
	max_iters_set = [10, 20, 30]
	Kvals = [8, 16, 24]
	
	for filename in filenames:
		fig = plt.figure()
		for num_iters in max_iters_set:
			for k in Kvals:
				kMeansOnPixels(filename, k, num_iters)
		plt.close(fig)

	






