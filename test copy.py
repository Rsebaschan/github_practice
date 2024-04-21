import numpy as np
import matplotlib.pyplot as plt
from test_utils import test

from helpers import sample_data, load_data, standardize


#-----------------------------------------------------------------------------------------------

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************


   
    xt_x = np.dot(tx.T, tx)  # X_transpose * X
    
    #det계산  np.linalg.det 없이
    if xt_x.shape == (2, 2):
        det_xt_x = xt_x[0, 0] * xt_x[1, 1] - xt_x[0, 1] * xt_x[1, 0]
    
    #det를 이용하여 inverse_tx 계산
    inv_xt_x = np.array([[xt_x[1, 1], -xt_x[0, 1]],
                           [-xt_x[1, 0], xt_x[0, 0]]]) / det_xt_x
    
    # w = (tx_transpose * tx)inverse * tx_transpose *y 
    # w = (xt_x)inverse * (tx_transpose *y)
    # w = inv_xt_x    *   (tx_transpose *y)
    w = np.dot(inv_xt_x, np.dot(tx.T, y))
    
    predicted_value = np.dot(tx, w)
    #MSE = (test_value - predicted_value)^2 / len(test_value)
    MSE =  np.mean((y - predicted_value)**2)
    
    return w, MSE
    raise NotImplementedError

#------------------------------------------------------------------------------------------------

# load data.
height, weight, gender = load_data()

# build sampled x and y.
seed = 1
#print(height.shape)  #1.846  #(10000,0)
#print(weight.shape) #(10000,0)
y = np.expand_dims(gender, axis=1)
#print(y)
X = np.c_[height.reshape(-1), weight.reshape(-1)]
#print(weight.reshape(-1))
#print(X.shape) #(10000,2)
y, X = sample_data(y, X, seed, size_samples=200)
#print(y)
#print(X)
x, mean_x, std_x = standardize(X)

#print(x.shape) #(200,2)
#print(mean_x.shape) #(2,0)
#print(std_x.shape) #(2,0)
from plots import visualization


def least_square_classification_visualize(y, x):
    """Least square demo

    Args:
        y:  shape=(N, 1)
        x:  shape=(N, 2)
    """
    # classify the data by linear regression
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    # ***************************************************
    # INSERT YOUR CODE HERE
    # classify the data by linear regression: TODO
    # ***************************************************
    # w = least squares with respect to tx and y
    slice_tx =tx[:,1:3]
    print(slice_tx.shape) #(200,2)
    #print(slice_tx)  #(-1.2267, -8.021)
    print(tx.shape) #(200,3)
    print(tx.shape[0]) #200 # shape[0], shape[1]를 이용하여 전체 행의 갯수와 열의 갯수를 반환받을 수 있다.
    #print(tx) #(1.0 , -1.2267 , -8.021)
    w, MSE =least_squares(y,slice_tx)
    
    w = np.vstack([0.5,w])
    # visualize your classification.
    visualization(y, x, mean_x, std_x, w, "classification_by_least_square")


least_square_classification_visualize(y, x)



