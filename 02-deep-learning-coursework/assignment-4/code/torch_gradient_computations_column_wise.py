import torch

# assumes X has size d x tau, h0 has size m x 1, etc
def ComputeGradsWithTorch(X, y, h0, RNN):

    tau = X.shape[1]

    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)


    ## give informative names to these torch classes        
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0) 
    
    W = torch_network['W']
    U = torch_network['U']
    b = torch_network['b']

    # Collect hidden states in a Python list (cleaner autograd graph than
    # in-place assignment into a pre-allocated torch.empty tensor).
    h_list = []
    hprev = ht
    for t in range(tau):
        #### BEGIN your code ######
        a_t = W @ hprev + U @ Xt[:, t:t + 1] + b   # (m, 1)
        h_t = apply_tanh(a_t)                       # (m, 1)
        h_list.append(h_t)
        hprev = h_t
        #### END of your code ######            

    Hs = torch.cat(h_list, dim=1)                   # (m, tau)

    Os = torch.matmul(torch_network['V'], Hs) + torch_network['c']
    P = apply_softmax(Os)

    # compute the loss
    loss = torch.mean(-torch.log(P[y, torch.arange(tau)]))

    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads
