import torch
import numpy as np

def ComputeGradsWithTorch_mlce(X, Y, network_params, lam):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    scores = torch.matmul(W, Xt)  + b;

    ## give an informative name to this torch class
    # apply_softmax = torch.nn.Softmax(dim=0)
    apply_sigmoid = torch.nn.Sigmoid()

    # apply softmax to each column of scores
    # P = apply_softmax(scores)
    P = apply_sigmoid(scores)
    
    ## compute the loss, cost
    # Stabilize BCE to avoid log(0)
    eps = 1e-12
    P = torch.clamp(P, eps, 1 - eps)
    loss = torch.mean(-((1 - Yt) * torch.log(1 - P) + Yt * torch.log(P)))
    cost = loss + lam * torch.sum(torch.multiply(W,W))

    # compute the backward pass relative to the cost and the named parameters 
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads    
