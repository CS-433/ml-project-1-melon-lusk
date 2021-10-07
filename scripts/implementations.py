import numpy as np


### COST FUNCTIONS

def compute_MSE(y,tx,w):
    e = y - np.dot(tx, w)
    loss = 1/(2*tx.shape[0])
    loss = loss * np.dot(e.T, e)
    return loss

def compute_MAE(y,tx,w):
    N = tx.shape[0]
    return (np.absolute(y - np.dot(tx,w)).sum())/N


def compute_loss(y, tx, w, kind = 'mse'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    if kind == 'mse':
        return compute_MSE(y,tx,w)
    elif kind == 'mae':
        return compute_MAE(y,tx,w)
    
### GRADIENT DESCENT
def compute_gradient(y, tx, w, kind = 'mse'):
    """Compute the gradient."""
    e = y - np.dot(tx,w)
    grad = (-1/tx.shape[0])*np.dot(tx.T,e) 
    loss = compute_loss(y,tx,w,kind)
    return (grad,loss)


def general_gradient_descent(y, tx, initial_w, max_iters, gamma, kind = 'mse'):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad,loss = compute_gradient(y,tx,w, kind)
        w = w - gamma * grad  
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return (w,loss)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return general_gradient_descent(y,tx,initial_w,max_iters,gamma,'mse')


### STOCHASTIC GRADIENT DESCENT
def compute_stoch_gradient(y, tx, w, kind = 'mse'):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    return compute_gradient(y,tx,w,kind)


def general_stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, kind = 'mse'):
    """Stochastic gradient descent algorithm."""
    losses, ws = [], []
    w = initial_w
    for i in range(max_iters):
        batches = batch_iter(y,tx,batch_size, 1)
        for batch in batches:
            grad, loss = compute_stoch_gradient(batch[0], batch[1], w, kind)
            w = w - gamma * grad
    return (w,loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    return general_stochastic_gradient_descent(y,tx,initial_w,len(y), max_iters,gamma, 'mse')


### LEAST SQUARES USING NORMAL EQUATIONS

def least_squares(y, tx):
    gram_matrix = np.dot(tx,tx.T)
    if np.linalg.det(gram_matrix) != 0:
        w = np.dot(np.linalg.inv(gram_matrix) , np.dot(tx, y))
    else:
        w = np.linalg.solve(gram_matrix, np.dot(tx,y))
    loss = compute_loss(y,tx,w,'mse')
    return (w, loss)
        
        
def ridge_regression(y,tx,lambda_):
        return np.dot((np.dot(tx,(tx.T)) + lambda_/(2*len(y)) * np.identity(tx.shape[0])) , np.dot(tx,y))
        
        
### LOGISTIC REGRESSION
        
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad, loss = compute_grad_logistic_regression(y,tx,w)
        w = w - gamma * grad
    return (w, loss)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]