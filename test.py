import numpy as np
import matplotlib.pyplot as plt


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


asdf = least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
print(asdf)









