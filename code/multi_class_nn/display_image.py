"""
Draw images of digits using python
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_bfgs, fmin

if __name__ == '__main__':
	np.random.seed(99999)
	mat_dict = loadmat('ex3data1.mat')
	X = mat_dict['X']
	y = mat_dict['y']
	m = X.shape[0]

	num_sample = 100
	sel = np.random.choice(range(0, m), size=num_sample, replace=False)

	