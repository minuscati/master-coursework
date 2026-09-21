import numpy as np
import copy
from model import ApplyNetwork, ComputeLoss, BackwardPass
import copy

def MiniBatchGD(X, Y, GDparams, init_net, lam):
    n = X.shape[1]
    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['epochs']
   
    trained_net = copy.deepcopy(init_net)
    Loss_list_train = []
    Loss_list_val = []
    Cost_list_train = []
    Cost_list_val = []

    # epoch loop
    for epoch in range(n_epochs):
        # shuffle training data 
        rng = np.random.default_rng(epoch)  
        perm = rng.permutation(n)
        X_tr = X[:, perm]
        Y_tr = Y[:, perm]
        Loss_sum = 0
        Cost_sum = 0
        n_step = n // n_batch

        # batch loop
        for j in range(n_step):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            inds = range(j_start, j_end)
            Xbatch = X_tr[:,inds]
            Ybatch = Y_tr[:,inds]
            ybatch = np.argmax(Ybatch, axis=0)
            # forward
            Pbatch = ApplyNetwork(Xbatch, trained_net)
            Lbatch = ComputeLoss(Pbatch, ybatch, trained_net)
            Cost = Lbatch + lam*np.sum(trained_net['W']**2)
            # backward
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # update
            trained_net['W'] = trained_net['W'] - eta * grads['W']
            trained_net['b'] = trained_net['b'] - eta * grads['b']
            Loss_sum += Lbatch
            Cost_sum += Cost

        # compute loss and cost on training data
        Loss_avg_train = Loss_sum / n_step
        Loss_list_train.append(Loss_avg_train)
        Cost_avg_train = Cost_sum / n_step
        Cost_list_train.append(Cost_avg_train)
        # compute loss on validation data
        P_val = ApplyNetwork(X_val, trained_net)
        L_val = ComputeLoss(P_val, y_val, trained_net)
        Cost_val = L_val + lam*np.sum(trained_net['W']**2)
        Loss_list_val.append(L_val)
        Cost_list_val.append(Cost_val)
    
    return trained_net, Loss_list_train, Loss_list_val, Cost_list_train, Cost_list_val

