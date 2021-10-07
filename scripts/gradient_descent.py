# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *
import numpy as np

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
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return general_gradient_descent(y,tx,initial_w,max_iters,gamma,'mse')