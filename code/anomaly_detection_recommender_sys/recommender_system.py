# -*- coding: utf-8 -*-97
""" Machine Learning Online Class
    Exercise 8 | Anomaly Detection and Collaborative Filtering
    Part 2: Recommender Systems
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b

def cofiCostFuncNonVectorized(params, Y, R, num_users, num_movies,
                              num_features, _lambda):
	""" Collaborative filtering cost function
		Non-vectorized implementation

	Parameters
	-----------------
	params : numpy.ndarray
	    Unroll vector of X and Theta
	Y : data matrix of rating
	R : relevance matrix
	_lambda : regularization parameter

	Return
	------------------
	J : collaborative filtering cost function
	"""
	X = params[0:(num_movies*num_features)].reshape(num_movies,
	                                                num_features)

	Theta = params[(num_movies*num_features):].reshape(num_users,
                                                           num_features)

	J = 0
	for i in xrange(num_movies):
		for j in xrange(num_users):
			if R[i,j] == 1:
				J += ( np.sum(Theta[j,:] * X[i,:]) - Y[i,j] ) ** 2
	J = J/2
	J += _lambda/2 * ( np.sum(Theta ** 2) + np.sum(X ** 2) )

	return J

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda):
	""" Collaborative filtering cost function
	    Vectorized implementation
	"""
	X = params[0:(num_movies*num_features)].reshape(num_movies,
	                                                num_features)

	Theta = params[(num_movies*num_features):].reshape(num_users,
                                                           num_features)

	J = np.sum( ((np.dot(X, Theta.T) - Y)  * R) ** 2 )/2
	J += _lambda/2 * ( np.sum(Theta ** 2) + np.sum(X ** 2) )

	return J


def cofiGradNonVectorized(params, Y, R, num_users, num_movies,
                          num_features, _lambda):
	""" Collaborative filtering gradient
	"""

	X = params[0:(num_movies*num_features)].reshape(num_movies,
	                                                num_features)

	Theta = params[(num_movies*num_features):].reshape(num_users,
		                                               num_features)

	X_grad = np.zeros(X.shape)
	Theta_grad = np.zeros(Theta.shape)

	for i in xrange(num_movies):
		for k in xrange(num_features):
			X_grad[i,k] = 0
			for j in xrange(num_users):
				if R[i,j] == 1:
					X_grad[i,k] += ( (np.sum(Theta[j,:] * X[i,:]) 
						             - Y[i,j]) * Theta[j,k] )

	for i in xrange(num_movies):
		X_grad[i,:] += _lambda * X[i,:]

	for j in xrange(num_users):
		for k in xrange(num_features):
			Theta_grad[j,k] = 0
			for i in xrange(num_movies):
				if R[i,j] == 1:
					Theta_grad[j,k] += ( (np.sum(Theta[j,:] * X[i,:]) 
						                - Y[i,j]) * X[i,k] )

	for j in xrange(num_users):
		Theta_grad[j,:] += _lambda * Theta[j,:]

	return np.r_['0', X_grad.flatten(), Theta_grad.flatten()]

def cofiGrad(params, Y, R, num_users, num_movies, num_features, _lambda):
	""" Collaborative filtering gradient
	    Vectorized Implementation
	"""

	X = params[0:(num_movies*num_features)].reshape(num_movies,
	                                                num_features)

	Theta = params[(num_movies*num_features):].reshape(num_users,
                                                           num_features)

	X_grad = np.zeros(X.shape)
	Theta_grad = np.zeros(Theta.shape)

	for i in xrange(num_movies):
		idx = (R[i,:] == 1).nonzero()[0]
		Theta_temp = Theta[idx,:]
		Y_temp = Y[i,idx]
		X_grad[i,:] = ( np.dot((np.dot(np.r_['0,2',X[i,:]], 
			                  Theta_temp.T) - Y_temp ),
			                  Theta_temp ) + _lambda * X[i,:] )

	for j in xrange(num_users):
		idx = (R[:,j] == 1).nonzero()[0]
		X_temp = X[idx,:]
		Y_temp = Y[idx,j]
		Theta_grad[j,:] = (np.dot((np.dot(X_temp, Theta[j,:].T) - Y_temp),
		                          X_temp ) + _lambda * Theta[j,:] )

	return np.r_['0', X_grad.flatten(), Theta_grad.flatten()]


def compute_numerical_gradient(J, theta):
    """ Tính numerical gradient theo công thức trên
    
    Parameters
    --------------
    J : wrapper function cho cost function. Hàm này nhận đối số là tham số theta
    theta: numpy.ndarray
        unrolled vector chứa các tham số của mạng neural
        
    Returns
    -------------
    numgrad : numpy.ndarray
       Giá trị của hàm gradient (numerical version)
    """
    numgrad = np.zeros( theta.shape )
    perturb = np.zeros( theta.shape )
    e = 1e-4

    for p in range( len(theta) ):
    	perturb[p] = e
    	# Set perturbation vector
    	perturb[p] = e
    	loss1 = J(theta - perturb)
    	loss2 = J(theta + perturb)
    	# Compute numerical gradient
    	numgrad[p] = (loss2 - loss1) / (2*e)
    	perturb[p] = 0

    return numgrad

def checkGradients(_lambda=0):
	""" Check collaborative filtering gradients
	    by comparing analytic gradients and numerical gradients
	"""

	# Create small problem
	X_t = np.random.random((4,3))
	Theta_t = np.random.random((5, 3))

	# Zap out most entries
	Y = np.dot(X_t, Theta_t.T)
	Y[np.random.random(Y.shape) > 0.5] = 0
	R = np.zeros(Y.shape)
	R[ Y != 0 ] = 1

	# Run Gradient Checking
	X = np.random.randn(*X_t.shape)
	Theta = np.random.randn(*Theta_t.shape)
	num_users = Y.shape[1]
	num_movies = Y.shape[0]
	num_features = Theta_t.shape[1]

	params = np.r_['0', X.flatten(), Theta.flatten()]

	costFunc = lambda p: cofiCostFunc(p, Y, R, num_users,
	                                  num_movies, num_features, _lambda)

	numgrad  = compute_numerical_gradient(costFunc, params)
	grad     = cofiGrad(params, Y, R, num_users,
	                    num_movies, num_features, _lambda)

	for ngr, gr in zip(numgrad, grad):
		print '\t%f  %f' % (ngr, gr)

	print('The above two columns you get should be very similar.\n'
		  '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

	diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
	
	print 'If your backpropagation implementation is correct, then'
	print 'the relative difference will be small (less than 1e-9).'
	print 'Relative Difference: %g' % diff

def loadMovieList():
	""" Load movie list into a list
	"""
	movie_list = []
	f = open('movie_ids.txt', 'rU')
	for line in f:
		line = line.strip()
		if line == '': continue
		idx, movie_name = line.split(' ', 1)
		movie_list.append(movie_name)
	f.close()

	return movie_list

def normalizeRatings(Y, R):
	# Preprocess data by subtracting mean rating for every 
	# movie (every row)
	# [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each 
	# movie has a rating of 0 on average, and returns the mean 
	# rating in Ymean

	m, n = Y.shape
	Ymean = np.zeros(m)
	Ynorm = np.zeros(Y.shape)

	for i in xrange(m):
		idx = [ j for j in xrange(R[i,:].size) if R[i,j] == 1 ]
		Ymean[i] = np.mean(Y[i, idx])
		Ynorm[i, idx] = Y[i, idx] - Ymean[i]

	return Ynorm, Ymean

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')

	np.random.seed(99999)
	# ============= Part 1: Loading movie ratings dataset ===========
	# You will start by loading the movie ratings dataset to understand the structure of the data.

	print('Loading movie ratings dataset.\n')
	# Load data
	data = loadmat('ex8_movies.mat')
	Y = data['Y']
	R = data['R']
	print 'Shape of Y and R'
	print Y.shape
	print R.shape

	# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies 
	# on 943 users
	# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j
	# gave a rating to movie i

	# From the matrix, we can compute statistics like average rating.
	print('\nAverage rating for movie 1 (Toy Story): %f / 5\n' %
		  np.mean( Y[0, R[0, :] == 1] ) )
	
	# We can "visualize" the ratings matrix by plotting it with imagesc
	fig = plt.figure()
	plt.imshow(Y)
	plt.ylabel('Movies')
	plt.xlabel('Users')
	plt.draw()
	plt.pause(0.01)
	raw_input('Program paused. Press enter to continue.\n')
	plt.close(fig)

	# ====== Part 2: Collaborative Filtering Cost Function =======
	# You will now implement the cost function for collaborative filtering. To help you debug your cost function, we have included set of weights that we trained on that. Specifically, you should complete the code in cofiCostFunc.m to return J.

	# Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
	pdata = loadmat('ex8_movieParams.mat')
	X = pdata['X']
	Theta = pdata['Theta']
	num_users = pdata['num_users']
	num_movies = pdata['num_movies']

	# Reduce the data set size so that this runs faster
	num_users = 4; num_movies = 5; num_features = 3;
	X = X[0:num_movies, 0:num_features]
	Theta = Theta[0:num_users, 0:num_features]
	Y = Y[0:num_movies, 0:num_users]
	R = R[0:num_movies, 0:num_users]

	# Evaluate cost function
	params = np.r_['0', X.flatten(), Theta.flatten()]
	# J = cofiCostFuncNonVectorized(params, Y, R, num_users, 
	# 	            num_movies, num_features, 0)
	J = cofiCostFunc(params, Y, R, num_users, 
	             num_movies, num_features, 0)
	
	print('Cost at loaded parameters: %f\n'
	 	  '(this value should be about 22.22)' % J)

	raw_input('\nProgram paused. Press enter to continue.\n')

	# ====== Part 3: Collaborative Filtering Gradient ========
	# Once your cost function matches up with ours, you should now 
	# implement the collaborative filtering gradient function. 
	# Specifically, you should complete the code in 
	# cofiCostFunc.m to return the grad argument.

	print('\nChecking Gradients (without regularization) ... ')
	# Check gradients by running checkNNGradients
	checkGradients()
	raw_input('\nProgram paused. Press enter to continue.\n')

	# ==== Part 4: Collaborative Filtering Cost Regularization ======
	# Evaluate cost function

	J = cofiCostFuncNonVectorized(params, Y, R, num_users, 
		            num_movies, num_features, 1.5)
	print('Cost at loaded parameters (lambda = 1.5): %f '
         '\n(this value should be about 31.34)' % J)
	raw_input('\nProgram paused. Press enter to continue.\n')


	# === Part 5: Collaborative Filtering Gradient Regularization ====
	print('\nChecking Gradients (with regularization) ... ')

	checkGradients(1.5)
	raw_input('\nProgram paused. Press enter to continue.\n')

	# ========== Part 6: Entering ratings for a new user =============
	# Before we will train the collaborative filtering model, we will
	# first add ratings that correspond to a new user that we just 
	# observed. This part of the code will also allow you to put in 
	# your own ratings for the movies in our dataset!

	movieList = loadMovieList()

	# Initialize my ratings
	my_ratings = np.zeros( len(movieList) )

	# Check the file movie_idx.txt for id of each movie in our dataset
	# For example, Toy Story (1995) has ID 1, so to rate it "4", 
	# you can set (in Python index starts from 0)
	my_ratings[0]  = 4

	# Or suppose did not enjoy Silence of the Lambs (1991), you can set
	my_ratings[97] = 2

	# We have selected a few movies we liked / did not like and the
	# ratings we gave are as follows:

	my_ratings[6]   = 3
	my_ratings[11]  = 5
	my_ratings[53]  = 4
	my_ratings[63]  = 5
	my_ratings[65]  = 3
	my_ratings[68]  = 5
	my_ratings[182] = 4
	my_ratings[225] = 5
	my_ratings[354] = 5

	print('\nNew user ratings:')
	for i in xrange(len(my_ratings)):
		if my_ratings[i] > 0:
			print('Rated %d for %s' % (my_ratings[i], movieList[i]))

	raw_input('\nProgram paused. Press enter to continue.\n')

	# ======== Part 7: Learning Movie Ratings ============
	# Now, you will train the collaborative filtering model on a 
	# movie rating dataset of 1682 movies and 943 users
	
	print('\nTraining collaborative filtering...')

	# Load data
	movie_data = loadmat('ex8_movies.mat')
	Y = movie_data['Y']
	R = movie_data['R']

	# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies
	# by 943 users
	# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j 
	# gave a rating to movie i
	print 'Shape of Y and R'
	print Y.shape, R.shape; print

	# Add our own ratings to the data matrix
	
	Y = np.append(my_ratings.reshape(len(my_ratings),1), Y, axis=1)
	R = np.append((my_ratings != 0).reshape(len(my_ratings),1), 
		          R, axis=1)

	# Normalize Ratings
	Ynorm, Ymean = normalizeRatings(Y, R)

	# Useful Values
	num_movies   = Y.shape[0]
	num_users    = Y.shape[1]
	num_features = 10

	# Set Initial Parameters (Theta, X)
	X = np.random.randn(num_movies, num_features)
	Theta = np.random.randn(num_users, num_features)
	initial_parameters = np.r_['0', X.flatten(), Theta.flatten()]

	# Set Regularization
	_lambda = 10;
	theta = fmin_l_bfgs_b(cofiCostFunc, initial_parameters,
		              fprime=cofiGrad,
		              args=(Y, R, num_users, num_movies,
		                    num_features, _lambda),
		              maxiter=100, disp=False)[0]

	
	# Unfold the returned theta back into U and W
	X = theta[0:num_movies*num_features].reshape(num_movies, 
		                                         num_features)
	Theta = theta[num_movies*num_features:].reshape(num_users, 
		                                            num_features)

	print('Recommender system learning completed.')
	raw_input('\nProgram paused. Press enter to continue.\n')

	# ========= Part 8: Recommendation for you ==============
	# After training the model, you can now make recommendations 
	# by computing the predictions matrix.

	p = np.dot(X, Theta.T) # num_movies x num_users
	my_predictions = p[:,0] + Ymean

	movieList = loadMovieList()
	ix = sorted( range(len(my_predictions)), 
		         key=lambda k: my_predictions[k], reverse=True )

	print('\nTop recommendations for you:')

	for i in range(10):
		j = ix[i]
		print('Predicting rating %.1f for movie %s' 
		      % (my_predictions[j], movieList[j]))

	print('\nOriginal ratings provided:')
	for i in range(len(my_ratings)):
		if my_ratings[i] > 0:
			print('Rated %d for %s' % (my_ratings[i], movieList[i]))
