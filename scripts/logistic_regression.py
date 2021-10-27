import numpy as np

#Normal logistic
def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx @ w)
    loss = 1/y.shape[0] * (y.T @ np.log(pred + 1e-10) + (1 - y).T @ np.log(1 - pred + 1e-10))
    return np.squeeze(-loss)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx @ w)
    return 1/y.shape[0] * (tx.T @ (pred - y))

def calculate_hessian(tx, w) :
    pred = sigmoid(tx @ w)
    txTD = np.einsum('ij,j->ij', tx.T, pred * (1 - pred))
    
    return txTD @ tx


def grad_loss(y, tx, w):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    return grad, loss


#Penalized logistic

def penalized_grad_loss(y, tx, w, lambda_):
    loss, grad = calculate_loss(y,tx,w), calculate_gradient(y,tx,w)
    loss += (lambda_/2) * np.power(np.linalg.norm(w),2)
    grad += lambda_ * w
    return grad, loss

def penalized_logistic_regression(y,tx,initial_w, max_iters, gamma, lambda_):
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

#Logistic regression (using Newton's method)
def logistic_regression(y,tx,initial_w, max_iters, gamma):
    ws , losses = [initial_w], []
    w = initial_w
    for i in range(max_iters):
        grad, loss = grad_loss(y,tx,w)
        hess = calculate_hessian(tx, w)
        w = w - gamma * np.linalg.solve(hess, grad)
        #w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Logistic Regression ({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=i, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss
    