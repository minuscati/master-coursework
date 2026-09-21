# Assignment 1
from fileinput import filename
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from torch_gradient_computations_v1 import ComputeGradsWithTorch1
from torch_gradient_computations_v2 import ComputeGradsWithTorch2
import copy
import torch

def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding = 'bytes')

    X = data_dict[b'data'].astype(np.float64)/255.0 
    X = X.transpose() # d*n(3072*10000)

    y = data_dict[b'labels'] # n*1

    # one-hot encoding of the labels
    Y = pd.get_dummies(y).values.astype(np.float64)
    Y = Y.transpose() # K*n(10*10000)
    return X, Y ,y

def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']
    s = W @ X + b
    s_stable = s - np.max(s, axis=0)
    P = np.exp(s_stable) / np.sum(np.exp(s_stable), axis=0)
    return P

def ComputeLoss(P, y):
    n = P.shape[1] 
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
            Lbatch = ComputeLoss(Pbatch, ybatch)
            Cost = Lbatch + lam*np.sum(trained_net['W']**2)
            # backward
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # update
            trained_net['W'] = trained_net['W'] - eta * grads['W']
            trained_net['b'] = trained_net['b'] - eta * grads['b']
            Loss_sum += Lbatch
            Cost_sum += Cost

        # compute loss and cost on training data
        P_tr_final = ApplyNetwork(X_tr, trained_net)
        Loss_avg_train = ComputeLoss(P_tr_final, np.argmax(Y_tr, axis=0))
        Cost_avg_train = Loss_avg_train + lam*np.sum(trained_net['W']**2)

        Loss_list_train.append(Loss_avg_train)
        Cost_list_train.append(Cost_avg_train)
        # compute loss on validation data
        P_val = ApplyNetwork(X_val, trained_net)
        L_val = ComputeLoss(P_val, y_val)
        Cost_val = L_val + lam*np.sum(trained_net['W']**2)
        Loss_list_val.append(L_val)
        Cost_list_val.append(Cost_val)
    
    return trained_net, Loss_list_train, Loss_list_val, Cost_list_train, Cost_list_val


def ComputeGradsWithTorch2(X, y, network_params, lam):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    N = X.shape[1]
    
    scores = torch.matmul(W, Xt)  + b;

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss, cost
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    
    cost = loss + lam * torch.sum(torch.multiply(W,W))

    # compute the backward pass relative to the cost and the named parameters 
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads    

    """
    Plot histogram of the probability assigned to the ground-truth class
    for correctly and incorrectly classified test examples.
    """
    y = np.asarray(y).reshape(-1).astype(int)
    n = P.shape[1]
    assert y.shape[0] == n, "y and P must correspond to the same test set size"

    # Predicted class for each test example
    y_pred = np.argmax(P, axis=0)
    correct_mask = (y_pred == y)

    # Probability of the ground-truth class for each test example:
    # P has shape (K, n), so gather P[y_i, i]
    p_true = P[y, np.arange(n)]
    p_true_correct = p_true[correct_mask]
    p_true_wrong = p_true[~correct_mask]

    plt.figure(figsize=(7, 4))
    plt.hist(p_true_correct, bins=bins, alpha=0.6, label='Correctly classified', color='green')
    plt.hist(p_true_wrong, bins=bins, alpha=0.6, label='Incorrectly classified', color='red')
    plt.xlabel('Probability assigned to ground-truth class')
    plt.ylabel('Number of test examples')
    if title is None:
        title = 'Histogram of ground-truth class probabilities'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
#--------------------------------------------------- main function ---------------------------------------------------
# Step 1: read in the training, validation and test data
cifar_dir = 'D:\\APPDATA\\Coding\\DL_course\\Datasets\\cifar-10-batches-py'
X_tr, Y_tr, y_tr = LoadBatch(cifar_dir + '\\data_batch_1')
X_val, Y_val, y_val = LoadBatch(cifar_dir + '\\data_batch_2')
X_te, Y_te, y_te = LoadBatch(cifar_dir + '\\test_batch')


# Step 2: normalize the train, validation and test data
[d, n] = X_tr.shape
K = Y_tr.shape[0]
mean_X = np.mean(X_tr, axis=1).reshape(d,1)
std_X = np.std(X_tr, axis=1).reshape(d,1)

X_tr = (X_tr-mean_X)/std_X # normalized train data
X_val = (X_val-mean_X)/std_X # normalized validation data
X_te = (X_te-mean_X)/std_X # normalized test data


# Step 3: Initialize W,b
rng = np.random.default_rng()
BitGen = type(rng.bit_generator)
# seed =25
seed = 42
rng.bit_generator.state = BitGen(seed).state
init_net = {}
init_net['W'] = .01*rng.standard_normal(size = (K,d))
init_net['b'] = np.zeros((K,1))


# Step 4: Apply network
P = ApplyNetwork(X_tr[:], init_net)


# Step 5: compute the loss function
L = ComputeLoss(P, y_tr)
print(f"Loss: {L}")


# Step 6: compute the accuracy
acc = ComputeAccuracy(P, y_tr)
print(f"Accuracy: {acc}")


# Step 7: evaluate the gradients
lam = 0.1
grads = BackwardPass(X_tr, Y_tr, P, init_net, lam)
torch_grads = ComputeGradsWithTorch2(X_tr, y_tr, init_net, lam)

# snipped check
d_small = 10
n_small = 3
small_net = {}
small_net['W'] =  .01*rng.standard_normal(size = (10, d_small))
small_net['b'] = np.zeros((10,1))

X_small = X_tr[0:d_small, 0:d_small]
Y_small = Y_tr[:, 0:d_small]
y_small = y_tr[0:d_small]

P = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, P, small_net, lam)
# tor_grads = ComputeGradsWithTorch1(X_small, y_small, small_net)
tor_grads = ComputeGradsWithTorch2(X_small, y_small, small_net, lam)

## absolute difference check
epi = 1e-6
ResultSame = True
for key in my_grads:
    abs_diff = np.abs(my_grads[key] - tor_grads[key])
    max_diff = np.max(abs_diff)
    print(f"Absolute difference of {key}: {max_diff}")

    if not np.all(abs_diff < epi):
        print(f"{key} gradients differ!")
        ResultSame = False

if ResultSame:
    print("Two methods produce the same gradients")
else:
    print("Two methods produce different gradients")

## relative error check
epi = 1e-6
ResultSame = True
for key in my_grads:
    denom = np.maximum(epi, (np.abs(tor_grads[key])+np.abs(my_grads[key])))
    rela_error = np.abs(tor_grads[key]-my_grads[key])/ denom
    max_err = np.max(rela_error)
    print(f"Relative error of {key}: {max_err}")
    if not np.all(rela_error < epi):
        print(f"{key} gradients differ!")
        ResultSame = False

if ResultSame:
    print("Two methods produce the same gradients")
else:
    print("Two methods produce different gradients")


# step 8:perform mini-batch gradient descent
n_batch = 100
eta = .001
n_epochs = 40
# lam = 0.1
GDparams = {'n_batch': n_batch, 'eta': eta, 'epochs': n_epochs}
trained_net, Loss_list_train, Loss_list_val, Cost_list_train, Cost_list_val = MiniBatchGD(X_tr, Y_tr, GDparams, init_net, lam)

# separated loss and cost plot
# plt.figure(figsize=(14, 5))
# plt.subplot(1, 2, 1)
# plt.plot(Cost_list_train, color = 'green')
# plt.plot(Cost_list_val, color = 'red')
# plt.xlim(left=0)
# plt.legend(['training Cost', 'validation Cost'])
# plt.xlabel('epoch')
# plt.ylabel('cost')
# plt.title('Cost(lambda = 1, eta = 0.001)')

# plt.subplot(1, 2, 2)
# plt.plot(Loss_list_train, color = 'green', linestyle = '--')
# plt.plot(Loss_list_val, color = 'red', linestyle = '--')
# plt.xlim(left=0)
# plt.legend(['training Loss', 'validation Loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Loss(lambda = 1, eta = 0.001)')

# plt.tight_layout()
# plt.show()

# combined loss and cost plot
plt.plot(Cost_list_train, color = 'green')
plt.plot(Loss_list_train, color = 'green', linestyle = '--')

plt.plot(Cost_list_val, color = 'red')
plt.plot(Loss_list_val, color = 'red', linestyle = '--')

plt.xlim(left=0)
plt.legend(['training Cost', 'training Loss', 'validation Cost', 'validation Loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('lambda = 0, eta = 0.001')
# plt.savefig('Graph of Cost and Loss.png')
plt.show()

# compute accuracy on test data
P_te = ApplyNetwork(X_te, trained_net)
acc_te = ComputeAccuracy(P_te, y_te)
print(f"Accuracy on test data:{acc_te*100:.2f}%")

# visualize the weight matrix
Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order = 'F')
W_im = np.transpose(Ws, (1,0,2,3))

fig = plt.figure(figsize=(15, 3))
for i in range(10):
    w_im = W_im[:, :, :, i]
    w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
    plt.subplot(1, 10, i+1)
    plt.imshow(w_im_norm)
    plt.axis('off')

fig.suptitle('lambda = 1, eta = 0.001', fontsize=12, y=0.8) 
# plt.savefig('Visualization of weight matrix.png', bbox_inches='tight')
plt.show()


input("Press Enter to close...")


