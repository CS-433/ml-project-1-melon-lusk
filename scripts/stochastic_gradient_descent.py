# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from gradient_descent import *

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
            ws.append(w)
            losses.append(loss)
    return losses, ws


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    return general_stochastic_gradient_descent(y,tx,initial_w,len(y), max_iters,gamma, 'mse')