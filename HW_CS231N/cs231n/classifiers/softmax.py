from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        sum_p = p.sum()

        ds = np.zeros_like(scores) #初始化一个score相同的数组存储dp / ds
        for j in range(len(scores)):
            if j == y[i]:
                ds[j] = -(sum_p - p[y[i]]) / sum_p
            else:
                ds[j] = p[j] / sum_p

        p /= p.sum()  # normalize
        logp = np.log(p)
        loss -= logp[y[i]]  # negative log probability is the loss
        
        dW += X[i].reshape(X[i].shape[0],1)  @  ds.reshape(1,ds.shape[0])
    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2*reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs and outputs are the same as softmax_loss_naive.
    """
    #print("cal")
    # Initialize the loss and gradient to zero.
    #print('here')
    loss = 0.0
    dW = np.zeros_like(W)
    scores = X.dot(W)
    #print(scores.shape) #500 10
    max_scores = np.max(scores,axis = 1)
    #print(max_scores.shape) # 500,
    scores_norm = scores - max_scores.reshape(max_scores.shape[0],1)
    #print(scores_norm.shape) #500,10
    exp_score = np.exp(scores_norm)
    p = exp_score / np.sum(exp_score,axis = 1,keepdims=True)
    #print(p.shape) #500 10
    loss = np.sum(-np.log(p[np.arange(p.shape[0]),y])) / X.shape[0]
    #print(X.shape)
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    #p_yi = p[np.arange(p.shape[0]),y]
    ds = p.copy() #除正确类别点外，导数和概率相等
    #print(ds.shape)
    ds[np.arange(p.shape[0]),y] = -(np.sum(ds,axis=1) - ds[np.arange(p.shape[0]),y]) / np.sum(ds,axis=1)
    #print( ds[np.arange(p.shape[0]),y].shape)
    dW = X.T @ ds / X.shape[0] + 2*reg * W

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
