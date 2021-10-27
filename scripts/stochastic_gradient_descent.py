# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from gradient_descent import *
from implementations import batch_iter

def compute_stoch_gradient(y, tx, w, kind = 'mse'):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y,tx,w,kind)


def general_stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, kind = 'mse'):
    """Stochastic gradient descent algorithm."""
    losses, ws = [], []
    w = initial_w
    for n_iter in range(max_iters):
        batches = batch_iter(y,tx,batch_size, 1)
        for batch in batches:
            grad, loss = compute_stoch_gradient(batch[0], batch[1], w, kind)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)
            
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return losses, ws


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    return general_stochastic_gradient_descent(y,tx,initial_w,len(y), max_iters,gamma, 'mse')

def adaptive_step_stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, kind = 'mse'):
    """Stochastic gradient descent algorithm."""
    losses, ws = [], [initial_w]
    w = initial_w
    for n_iter in range(max_iters):
        batches = batch_iter(y,tx,batch_size, 1)
        for batch in batches:
            grad, loss = compute_stoch_gradient(batch[0], batch[1], w, kind)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)
            grad2,trash = compute_stoch_gradient(batch[0], batch[1],ws[len(ws)-1], kind)
            
            gamma=abs((np.dot(ws[len(ws)-1]-ws[len(ws)-2],grad-grad2)))/(np.dot(grad-grad2,grad-grad2))
            
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return losses, ws