"""
Draw images of digits using python
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import matplotlib.cm as cm
import math
from scipy.optimize import fmin_bfgs, fmin

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
		plt.draw()
		plt.pause(0.5)

	return display_array

if __name__ == '__main__':
	np.random.seed(99999)
	mat_dict = loadmat('ex3data1.mat')
	X = mat_dict['X']
	y = mat_dict['y']
	m = X.shape[0]

	num_sample = 100
	sel = np.random.choice(range(0, m), size=num_sample, replace=False)
	displayData(X[sorted(sel),:])
	raw_input('<Press Enter to continue>\n')



	