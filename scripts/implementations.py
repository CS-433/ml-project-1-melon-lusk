import numpy as np


## GRADED METHODS

### LEAST SQUARES GD

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return general_gradient_descent(y,tx,initial_w,max_iters,gamma,'mse')

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


### LEAST SQUARES SGD

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    return general_stochastic_gradient_descent(y,tx,initial_w,1, max_iters,gamma, 'mse')


def compute_stoch_gradient(y, tx, w, kind = 'mse'):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
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


### LEAST SQUARES USING NORMAL EQUATIONS

def least_squares(y, tx):
    gram_matrix = np.dot(tx.T,tx)
    w = np.linalg.solve(gram_matrix, np.dot(tx.T,y))
    loss = compute_loss(y,tx,w,'mse')
    return (w, loss)

### RIDGE REGRESSION USING NORMAL EQUATIONS

def ridge_regression(y,tx,lambda_):
        w =  np.linalg.solve((np.dot(tx.T,tx) + lambda_*(2*len(y)) * np.identity(tx.T.shape[0])) , np.dot(tx.T,y))
        loss = compute_loss(y,tx,w,'mse')
        return w, loss
    
    

    
### LOGISTIC REGRESSION

def calculate_gradient_MLE(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx @ w)
    return 1/y.shape[0] * (tx.T @ (pred - y))

def logistic_regression(y,tx,initial_w, max_iters, gamma):
    ws , losses = [initial_w], []
    w = initial_w
    for i in range(max_iters):
        loss = compute_loss(y, tx, w, 'mle')
        grad = calculate_gradient_MLE(y,tx,w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Logistic Regression ({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=i, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss

### REGULARIZED LOGISTIC REGRESSION

def penalized_grad_loss(y, tx, w, lambda_):
    loss, grad = compute_loss(y,tx,w, 'mle'), calculate_gradient_MLE(y,tx,w)
    loss += (lambda_/2) * np.power(np.linalg.norm(w),2)
    grad += lambda_ * w
    return grad, loss

def reg_logistic_regression(y,tx,initial_w, max_iters, gamma, lambda_):
    ws , losses = [initial_w], []
    w = initial_w
    for i in range(max_iters):
        grad, loss = penalized_grad_loss(y,tx,w, lambda_)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Logistic Regression ({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=i, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss


## HELPER METHODS

def compute_MSE(y,tx,w):
    e = y - np.dot(tx, w)
    loss = 1/(2*tx.shape[0])
    loss = loss * np.dot(e.T, e)
    return loss

def compute_MAE(y,tx,w):
    N = tx.shape[0]
    return (np.absolute(y - np.dot(tx,w)).sum())/N

def compute_MLE(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx @ w)
    loss = 1/y.shape[0] * (y.T @ np.log(pred + 1e-10) + (1 - y).T @ np.log(1 - pred + 1e-10))
    return np.squeeze(-loss)

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1 + np.exp(-t))


def compute_loss(y, tx, w, kind = 'mse'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    if kind == 'mse':
        return compute_MSE(y,tx,w)
    elif kind == 'mae':
        return compute_MAE(y,tx,w)
    elif kind == 'mle':
        return compute_MLE(y,tx,w)
        



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