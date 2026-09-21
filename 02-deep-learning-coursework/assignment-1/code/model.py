import numpy as np

def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']
    s = W @ X + b
    P = np.exp(s) / np.sum(np.exp(s), axis=0)
    return P

def ComputeLoss(P, y, network):
    n = P.shape[1] #10000
    W = network['W']
    L2_reg = np.sum(W**2)
    py = P[y, range(n)]
    l_ce = -np.log(py)
    Loss = np.sum(l_ce)/n
    return Loss

def ComputeAccuracy(P, y):
    n = P.shape[1]
    y_tr_pred = np.argmax(P, axis=0)
    acc = sum(y_tr_pred == y)/n
    return acc

def BackwardPass(X, Y, P, network, lam):
    n = P.shape[1]
    W = network['W']
    grad_W = (1/n) * (P-Y) @ X.T + 2*lam*W
    grad_b = (1/n) * np.sum(P-Y, axis=1).reshape(K,1)
    grads = {'W': grad_W, 'b': grad_b}
    return grads

