# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *
import numpy as np
from logistic_regression import *

def compute_gradient(y, tx, w, kind = 'mse'):
    """Compute the gradient."""
    e = y - np.dot(tx,w)
    if kind == 'mse' :
        grad = (-1/tx.shape[0])*np.dot(tx.T,e) 
    if kind == 'mae' :
        grad = (-1/tx.shape[0])* tx.T@np.sign(e)
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
        
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return losses, ws

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return general_gradient_descent(y,tx,initial_w,max_iters,gamma,'mse')


def adaptative_step_gradient_descent(y, tx, initial_w, max_iters, gamma, kind = 'mse'):
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
        grad2,trash =compute_gradient(y,tx,ws[-1], kind)
        
        gamma=abs((np.dot(ws[-1]-ws[-2],grad-grad2)))/(np.dot(grad-grad2,grad-grad2))
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss