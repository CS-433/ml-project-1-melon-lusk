# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np

def compute_gradient_custom(y, tx, w):
    """Compute the gradient."""
    e = y - tx@w
    grad = (-1/tx.shape[0])*np.dot(tx.T,e)
    return grad

def compute_MSE_custom(y, tx, w):
    e = y - tx @ w
    return 1/(2 * y.shape[0]) * e.T @ e

def compute_hessian(tx) :
    return 1/tx.shape[0] * tx.T @ tx


def hessian_descent(y, tx, initial_w, max_iters, gamma):
    
    ws = [initial_w]
    losses = []
    w = initial_w
    invB = np.identity(w.shape[0])
    grad = compute_gradient_custom(y,tx,w)
    for n_iter in range(max_iters):
        grad_prev = np.copy(grad)
        loss = compute_MSE_custom(y,tx,w)
        
        w = w - gamma * np.linalg.inv(compute_hessian(tx)) @ grad
            
        
        grad = compute_gradient_custom(y,tx,w)
        ws.append(w)
        
        losses.append(loss)
        
        print("Hessian Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss