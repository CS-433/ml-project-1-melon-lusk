# -*- coding: utf-8 -*-

# Custom method of gradient descent: we use the tanh function to force the output of the dot product
# between tx and w between -1 and 1, as a way to make the predicted output closer to the values that
# can appear in y (i.e -1 and 1).


import numpy as np

def compute_gradient_custom(y, tx, w):
    """Compute the gradient."""
    e = y - np.tanh(tx@w) #the error vector is the difference of the true output and the tanh of the dot product
    grad = (-1/tx.shape[0])*np.dot(tx.T,e)
    return grad

def compute_MSE_custom(y, tx, w):
    e = y - np.tanh(tx @ w) #we change the cost function to take into account the tanh
    return 1/(2 * y.shape[0]) * e.T @ e

def compute_hessian(tx) :
    return 1/tx.shape[0] * tx.T @ tx


def newton_method(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    invB = np.identity(w.shape[0])
    grad = compute_gradient_custom(y,tx,w)
    hess = compute_hessian(tx)
    
    for n_iter in range(max_iters):
        grad_prev = np.copy(grad)
        loss = compute_MSE_custom(y,tx,w)
        
        prev = np.linalg.solve(hess, grad)
        
        w = w - gamma * prev
            
        grad = compute_gradient_custom(y,tx,w)
        
        ws.append(w)
        
        losses.append(loss)
        
        curr = np.linalg.solve(hess, grad)
        gamma=abs((np.dot(ws[-1]-ws[-2],prev-curr)))/(np.dot(prev-curr,prev-curr))
        
        print("Hessian Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss