# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *
import numpy as np

def compute_gradient(y, tx, w, kind = 'mse'):
    """Compute the gradient."""
    e = y - np.dot(tx,w)
    if kind == 'mse' :
        grad = (-1/tx.shape[0])*np.dot(tx.T,e) 
    if kind == 'mae' :
        grad = (-1/tx.shape[0])* tx.T@np.sign(e)
    loss = compute_loss(y,tx,w,kind)
    return (grad,loss)

# ##General algorithm for gradient descent
def general_gradient_descent(y, tx, initial_w, max_iters, gamma, kind = 'mse'):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad,loss = compute_gradient(y,tx,w, kind)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return losses, ws

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return general_gradient_descent(y,tx,initial_w,max_iters,gamma,'mse')


def adaptative_step_gradient_descent(y, tx, initial_w, max_iters, gamma, kind = 'mse'):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad,loss = compute_gradient(y,tx,w, kind)
        w = w - gamma * grad
        
        ws.append(w)
        
        losses.append(loss)
        grad2,trash =compute_gradient(y,tx,ws[-1], kind)
        
        gamma=abs((np.dot(ws[-1]-ws[-2],grad-grad2)))/(np.dot(grad-grad2,grad-grad2)) # ## adaptive step; see the report for the formula
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss


### MAE
def compute_gradient_MAE(y, tx, w):
    e = y - np.dot(tx,w)
    grad = (-1/tx.shape[0])*(np.dot(tx.T,np.sign(e)))
    loss = compute_loss(y,tx,w,'mae')
    return (grad,loss)



def MAE_gradient_descent(y, tx, initial_w, max_iters, gamma ):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad,loss = compute_gradient_MAE(y,tx,w)
        w = w - gamma * grad  
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return (w,loss)