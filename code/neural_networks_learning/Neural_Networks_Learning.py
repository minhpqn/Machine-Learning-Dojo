# -*- coding: utf-8 -*-
""" Programming Exercise 4: Neural Network Learning

    Hướng dẫn cài đặt mạng neural networks bằng python
    Script dựa trên bài tập lập trình số 4 (Machine Learning Course)
    của giáo sư Andrew Ng.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def sigmoid(z):
    # z can be np.ndarray, np.matrix, or scalar
    return 1 / ( 1 + np.exp(-z) ) 

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                     num_labels, X, y, _lambda):
    """ Tính cost function cho neural networks
    
    Parameters
    -----------
    nn_params : ndarray
             unrolled vector of theta1 and theta2
    theta1 : ndarray
             Tham số của mạng neural cho tầng ẩn (tầng thứ 2)
    theta2 : ndarray
             Tham số của mạng neural cho tầng output (tầng thứ 3)
    input_layer_size : int
    hidden_layer_size : int
    num_labels : int
    X      : ndarray
             Ma trận trong đó mỗi hàng lưu các features của các examples
    y      : ndarray
             Danh sách nhãn của các examples
    _lambda : float
             Tham số trong công thức regularization    
    
    Returns
    -----------
    f : float
        Giá trị của cost function trong mạng neural
    
    Notes
    -----------
    Kích thước của ma trận trong dữ liệu được cung cấp
    theta1 :  25 x 401
    theta2 :  10 x 26
    X : 5000 x 400
    y : (5000, ) ndim = 1
    """    
    m = X.shape[0]
    theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape( 
                             (hidden_layer_size, input_layer_size + 1))    
    
    theta2 = nn_params[hidden_layer_size * (input_layer_size+1):].reshape(
                            (num_labels, hidden_layer_size+1))
        
    X_n = np.concatenate( ( np.ones((m,1)), X ), axis=1 )
    Z2 = np.dot( X_n, theta1.T ) # 5000 x 25
    A2 = sigmoid(Z2)
    A2 = np.concatenate( ( np.ones((m,1)), A2 ), axis=1 )
    Z3 = np.dot( A2, theta2.T ) # 5000 * 10
    A3 = sigmoid(Z3)
    
    # TASK: Cải tiến cài đặt dưới đây -- chỉ dùng 1 vòng lặp trên các examples
    J = 0    
    for i in xrange(m):
        for k in xrange(1, num_labels+1):
            yk = np.int(y[i]==k)
            cost = -yk * np.log( A3[i,k-1] ) - (1-yk) * np.log( 1 - A3[i,k-1] )
            J += cost
    J /= m
    
    # Regularized version of cost function 
    J += _lambda * ( (theta1[:,1:] ** 2).sum() + (theta2[:,1:] ** 2).sum() ) / (2*m)
    
    return J

def rand_initialize_weights(L_in, L_out):
    """ Khởi tạo ngẫu nhiên các tham số ban đầu cho một tầng
    Parameters
    -----------
    L_in : int
           Số connection đầu vào của tầng
    L_out : int
           Số connection đầu ra của tầng
           
    Return
    ----------
    W : np.float64
        Ma trận với kích thước L_out x (1 + L_in) gồm các số ngẫu nhiên trong khoảng
        [-epsilon, -epsilon]
    """
    epsilon_init = 0.12
    return ( np.random.randn(L_out, 1 + L_in) * 2 * epsilon_init - 
    	     epsilon_init )

def sigmoid_gradient(z):
    # z can be scalar or np.ndarray
    g = sigmoid(z)
    return g * (1-g)

def nn_gradient(nn_params, input_layer_size, hidden_layer_size,
                num_labels, X, y, _lambda):
    """ Tính gradient cho cost function trong neural networks
    
    Parameters
    -----------
    nn_params : ndarray
             unrolled vector of theta1 and theta2
    theta1 : ndarray
             Tham số của mạng neural cho tầng ẩn (tầng thứ 2)
    theta2 : ndarray
             Tham số của mạng neural cho tầng output (tầng thứ 3)
             
    input_layer_size : int
    hidden_layer_size : int
    num_labels : int
    
    X      : ndarray
             Ma trận trong đó mỗi hàng lưu các features của các examples
    y      : ndarray
             Danh sách nhãn của các examples
    _lambda : float
             Tham số trong công thức regularization    
    
    Returns
    -----------
    grad : ndarray
        unrolled vector của gradient cho mỗi tầng
    
    Notes
    -----------
    Kích thước của ma trận trong dữ liệu được cung cấp
    theta1 :  25 x 401
    theta2 :  10 x 26
    X : 5000 x 400
    y : (5000, ) ndim = 1
    """    
    
    m = X.shape[0]
    theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape( 
                             (hidden_layer_size, input_layer_size + 1))    
    
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
                            (num_labels, hidden_layer_size + 1))
    
    theta1_grad = np.zeros( theta1.shape )
    theta2_grad = np.zeros( theta2.shape )
        
    X_n = np.concatenate( ( np.ones((m,1)), X ), axis=1 )
    Z2 = np.dot( X_n, theta1.T ) # 5000 x 25
    A2 = sigmoid(Z2)
    # A2.shape now -> m x (1+hidden_layer_size)
    A2 = np.concatenate( ( np.ones((m,1)), A2 ), axis=1 )
    Z3 = np.dot( A2, theta2.T ) # 5000 * 10
    A3 = sigmoid(Z3)
    
    c = np.array( range(1, num_labels + 1) )
    for t in xrange(m):
        yt = y[t]
        # Step 1: Perform feedforward pass computing the activations
        # z1, a2, z3, a3 (Already done)
        a1 = X_n[[t],:] # 1 x 401 -- ugly indexing
        z2 = Z2[[t],:]
        a2 = A2[[t],:]
        z3 = Z3[[t],:]
        a3 = A3[[t],:] # shape = 1 x 10        
        
        # Step 2: Compute error terms for output layer
        # For each ouput unit k in layer 3 compute delta3[k]
        
        delta_3 = a3 - ( c == yt ) # shape 1 x 10
        
        # Step 3: Compute error terms for layer 2
        # delta_2 = np.dot(delta_3, theta2) * [ 1 sigmoid_gradient(z2) ]
        sigmoid_grad_z2 = np.concatenate((np.ones( (1,1) ), 
                                         sigmoid_gradient(z2) ),
                                         axis=1)
        delta_2 = np.dot(delta_3, theta2) * sigmoid_grad_z2 # shape = 1 x 26
        
        # Step 4: Accumulate the gradient from this example
        theta1_grad = theta1_grad + np.dot( delta_2[:,1:].T, a1 )
        theta2_grad = theta2_grad + np.dot( delta_3.T, a2 )
        
    theta1_grad /= m
    theta2_grad /= m
    # regularized version for gradient
    theta1_grad[:,1:] += _lambda * theta1[:,1:]/m
    theta2_grad[:,1:] += _lambda * theta2[:,1:]/m
    # unrolled gradients
    grad_ = np.concatenate( ( theta1_grad.flatten(), 
                             theta2_grad.flatten() ) )
      
    return grad_

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
        # Set perturbation vector
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
        
    return numgrad

def debugInitializeWeights(fan_out, fan_in):
    """
    initializes the weights 
    of a layer with fan_in incoming connections and fan_out outgoing 
    connections using a fix set of values
    Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    the first row of W handles the "bias" terms
    """
    W = np.zeros( (fan_out, 1 + fan_in) )
    W = np.sin( range(1, W.size+1) ).reshape(W.shape)/10
    return W  
    
def checkNNGradients(_lambda=0):
    """ Tạo một mạng neural cỡ nhỏ để kiểm tra backpropagation gradients
    Returns
    -------------
    None
    """
    
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    X = debugInitializeWeights(m, input_layer_size -1)
    y = 1 + np.mod( range(1,m+1), num_labels)
    nn_params = np.concatenate( ( theta1.flatten(), theta2.flatten()) )
    costFunc = lambda p: nn_cost_function(p, input_layer_size, 
                                          hidden_layer_size, num_labels,
                                          X, y, _lambda)
    
    gradFunc = lambda p: nn_gradient(p, input_layer_size, 
                                     hidden_layer_size, num_labels,
                                     X, y, _lambda)
    
    grad = gradFunc(nn_params)
    numgrad = compute_numerical_gradient(costFunc, nn_params)
      
    print 'The above two columns you get should be very similar.'
    print '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n'
    
    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used 
    # EPSILON = 0.0001 in compute_mumerical_gradient(),
    # then diff below should be less than 1e-9
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    
    print 'If your backpropagation implementation is correct, then'
    print 'the relative difference will be small (less than 1e-9).'
    print 'Relative Difference: %g' % diff

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X_new = np.concatenate( (np.ones( (m,1) ), X), axis=1 ) # 5000 x 401
    A2 = sigmoid( np.dot(X_new, Theta1.T) ) # 5000 x 25
    A2 = np.concatenate( ( np.ones( (m,1) ), A2 ), axis=1 ) # 5000 x 26
    A3 = sigmoid( np.dot(A2, Theta2.T) ) # 5000 x 10
    p = np.argmax( A3, axis=1 ) + 1
    return p.reshape( (m, 1) ) 

# Setup parameters for this exercise
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

# =========== Part 1: Loading and Visualizing Data =============
print 'Loading and Visualizing Data ...'
mat_dict = loadmat('ex4data1.mat')
X = mat_dict['X']
y = mat_dict['y']
m = X.shape[0]

# ================ Part 2: Loading Parameters ================
print '\nLoading Saved Neural Network Parameters ...'
para_dict = loadmat('ex4weights.mat')
Theta1 = para_dict['Theta1']
Theta2 = para_dict['Theta2']

nn_params = np.concatenate( (Theta1.flatten(), Theta2.flatten()) )

# ================ Part 3: Compute Cost (Feedforward) ================

print '\nFeedforward Using Neural Network ...'

_lambda = 0
J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                     num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights): %f \n'
      '(this value should be about 0.287629)' % J)

# =============== Part 4: Implement Regularization ===============

print '\nChecking Cost Function (with Regularization) ...'
_lambda = 1 
J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, 
	                 num_labels, X, y, _lambda)
print('Cost at parameters (loaded from ex4weights): %f \n'
      '(this value should be about 0.383770)' % J)

# ================ Part 5: Sigmoid Gradient  ================

print '\nEvaluating sigmoid gradient...'
for z in [10000, 0]:
    print 'sigmoid_gradient(%.0f) = %.4f' % (z, sigmoid_gradient(z))
    
g = sigmoid_gradient( np.array([1, -0.5, 0, 0.5, 1]))
print g

# ================ Part 6: Initializing Pameters ================

print '\nInitializing Neural Network Parameters ...'
np.random.seed(99999)
initial_Theta1 = rand_initialize_weights(input_layer_size, 
	                                     hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
# unroll parameters
initial_nn_params = np.concatenate( (initial_Theta1.flatten(), 
                                     initial_Theta2.flatten()) )


# =============== Part 7: Implement Backpropagation ===============

print '\nChecking Backpropagation...'
checkNNGradients()
print '\nChecking Backpropagation (w/ Regularization)'
_lambda = 3
checkNNGradients(_lambda)
debug_J = nn_cost_function(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, _lambda);

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f\n'
	  '\n(this value should be about 0.576051)\n' % debug_J);

# =================== Part 8: Training NN ===================
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b

_lambda = 1
print '\nTraining Neural Network...'
nn_params, f, d = fmin_l_bfgs_b(nn_cost_function, initial_nn_params,
	                  fprime=nn_gradient, 
	                  args=(input_layer_size, 
	                  	    hidden_layer_size, 
	                  	    num_labels, X, y, _lambda),
	                  disp=False, maxiter=50)

Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(
	                           (hidden_layer_size, input_layer_size + 1))    
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
                            (num_labels, hidden_layer_size + 1))


# ================= Part 10: Implement Predict =================
pred = predict(Theta1, Theta2, X)
print('\nTraining accuracy (with neural networks): %s' % 
      ( 100 * (pred == y).mean() ) )








