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
    return (


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
    gram_matrix = tx @ (tx.T)
    if np.linalg.det(gram_matrix) != 0:
        w = np.linalg.inv(gram_matrix) @ (tx @ y)
    else:
        w = np.linalg.solve(gram_matrix, tx@y)
    loss = compute_loss(y,tx,w,'mse')
    return (w, loss)
        
        
def ridge_regression(y,tx,lambda_):
        return (tx@(tx.T) + lambda_ * np.identity(tx.shape[0])) @ (tx@y)
        
        
### LOGISTIC REGRESSION
        
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad, loss = compute_grad_logistic_regression(y,tx,w)
        w = w - gamma * grad
    return (w, loss)