#Normal logistic
def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    dot_product = np.dot(tx, w)
    first_term = np.log( 1 + np.exp(dot_product))
    second_term = y * dot_product
    return np.sum(first_term - second_term, axis = 0)[0]


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(
        sigmoid(tx.dot(w)) -y  
    )


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    return grad, loss


#Penalized logistic

def penalized_logistic_regression(y, tx, w, lambda_):
    loss, grad = calculate_loss(y,tx,w), calculate_gradient(y,tx,w)
    loss += (lambda_/2) * np.power(np.linalg.norm(w),2)
    grad += lambda_ * w
    return grad, loss


def logistic_gradient_descent(y,tx,initial_w, max_iters, gamma,lambda_):
    ws , losses = [initial_w], []
    w = initial_w
    for i in range(max_iters):
        grad, loss = penalized_logistic_regression(y,tx,w,lambda_)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Loss at iteration ({i}/{m_i} : {loss})".format(i = i, m_i = max_iters, loss = loss))
    