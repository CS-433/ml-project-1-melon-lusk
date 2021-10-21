import numpy as np

#tanh method
def compute_loss(y, tx, w):
    """Calculate the loss."""
    e = y - np.tanh(np.dot(tx, w))
    loss = 1/(2*tx.shape[0])
    loss = loss * np.dot(e.T, e)
    return loss
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.tanh(np.dot(tx,w))
    grad = (-1/tx.shape[0])*np.dot(tx.T,e) 
    loss = compute_loss(y,tx,w)
    return (grad,loss)


def tanh_gradient_descent(y, tx, initial_w, max_iters, gamma): #adaptative step
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad,loss = compute_gradient(y,tx,w)
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        
        losses.append(loss)
        grad2,trash =compute_gradient(y,tx,ws[-1])
        
        gamma=abs((np.dot(ws[-1]-ws[-2],grad-grad2)))/(np.dot(grad-grad2,grad-grad2))
        
  
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, gamma={gamma}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], gamma=gamma))
    return w, loss


