# -*- coding: utf-8 -*-

import numpy as np
from logistic_regression import *

def compute_MSE(y,tx,w):
    e = y - np.dot(tx, w)
    loss = 1/(2*tx.shape[0])
    loss = loss * np.dot(e.T, e)
    return loss

def compute_MAE(y,tx,w):
    N = tx.shape[0]
    return (np.absolute(y - np.dot(tx,w)).sum())/N
"""Function used to compute the loss."""

def compute_loss(y, tx, w, kind = 'mse'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    if kind == 'mse':
        return compute_MSE(y,tx,w)
    elif kind == 'mae':
        return compute_MAE(y,tx,w)