import numpy as np
import matplotlib.pyplot as plt
from test_utils import test

from helpers import sample_data, load_data, standardize


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # 시그모이드 공식
    asd = 1 / (1 + np.exp(-t))
    return asd
    # ***************************************************
    
    
    raise NotImplementedError


test(sigmoid)

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    least square 부분 참고
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        y (=predicted_value) = np.dot(tx, w)
    
    
    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    #print(tx.shape[0]) # shape[0] 이용하여 전체 행의 갯수를 반환받을 수 있다.
                        # shape[1] 이용하여 전체 열의 갯수를 반환받을 수 있다.
    assert y.shape[0] == tx.shape[0]     #assert N == N
    assert tx.shape[1] == w.shape[0]     #assert D == D

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    # Calculate a as sigmoid of the dot product of tx and w
    a = sigmoid(tx.dot(w))
    
    # Loss 계산 by negative log likelihood(NLL) 은 Binary - cross Entropy 라고도 한다 
    loss = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / y.shape[0]
    
    return loss

    raise NotImplementedError



test(calculate_loss)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

        
    
    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    
    # a는 prediction value_y   
    a = sigmoid(tx.dot(w))
    # y는 실제값 a는 예측값
    error = y - a
    # Loss를 편미분w 한 값 = gradient  # gradient는 Loss의 weight에 대한 편미분으로 정의 
    gradient = -tx.T.dot(error) / y.shape[0]
    return gradient
    
    raise NotImplementedError("Calculate gradient")


test(calculate_gradient)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    a = sigmoid(tx.dot(w))
    # y는 실제값 a는 예측값
    error = y - a
    # Loss를 편미분w 한 값 = gradient  # gradient는 Loss의 weight에 대한 편미분으로 정의 
    gradient = -tx.T.dot(error) / y.shape[0]
    
    w = w - gamma * gradient
    
    loss = calculate_loss(y, tx, w)
   
    return loss, w
    
    raise NotImplementedError

# y = np.c_[[0., 1.]]
# tx = np.arange(6).reshape(2, 3)
# w = np.array([[0.1], [0.2], [0.3]])
# gamma = 0.1
# loss, w = learning_by_gradient_descent(y, tx, w, gamma)
# print(loss)
# print(w)

test(learning_by_gradient_descent)


def logistic_regression_gradient_descent_visualize(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.5
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(
        y,
        x,
        mean_x,
        std_x,
        w,
        "classification_by_logistic_regression_gradient_descent",
        True,
    )
    print("loss={l}".format(l=calculate_loss(y, tx, w)))




logistic_regression_gradient_descent_visualize(y, x)