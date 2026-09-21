# Assignment 1
# multiple binary cross-entropy loss
#eta = 0.001->0.01-> 0.1, acc= 36.77->38.10->33.68
from fileinput import filename
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_gradient_computations_v3 import ComputeGradsWithTorch_mlce
import copy

def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding = 'bytes')

    X = data_dict[b'data'].astype(np.float64)/255.0 # n*d(10000*3072)
    X = X.transpose() # d*n(3072*10000)

    y = data_dict[b'labels'] # n*1

    # one-hot encoding of the labels
    Y = pd.get_dummies(y).values.astype(np.float64)
    Y = Y.transpose() # K*n(10*10000)
    return X, Y ,y

def ApplyNetwork_sigmoid(X, network):
    W = network['W']
    b = network['b']
    s = W @ X + b
    P = 1 / (1+np.exp(-s))
    return P

def ComputeLoss_mlce(P, Y):
    eps = 1e-15
    P = np.clip(P, eps, 1 - eps)
    l_mlce = -((1-Y)*np.log(1-P)+Y*np.log(P))
    Loss = np.mean(l_mlce)
    return Loss

def ComputeAccuracy(P, y):
    n = P.shape[1]
    y_tr_pred = np.argmax(P, axis=0)
    acc = sum(y_tr_pred == y)/n
    return acc

def BackwardPass_mlce(X, Y, P, network, lam):
    n = P.shape[1]
    k = P.shape[0]
    W = network['W']
    grad_W = (1/(k*n)) * (P-Y) @ X.T + 2*lam*W
    grad_b = (1/(k*n)) * np.sum(P-Y, axis=1).reshape(k,1)   
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
            Pbatch = ApplyNetwork_sigmoid(Xbatch, trained_net)
            Lbatch = ComputeLoss_mlce(Pbatch, Ybatch)
            Cost = Lbatch + lam*np.sum(trained_net['W']**2)
            # backward
            grads = BackwardPass_mlce(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # update
            trained_net['W'] = trained_net['W'] - eta * grads['W']
            trained_net['b'] = trained_net['b'] - eta * grads['b']
            Loss_sum += Lbatch
            Cost_sum += Cost

        # compute loss and cost on training data
        # Loss_avg_train = Loss_sum / n_step
        # Loss_list_train.append(Loss_avg_train)
        # Cost_avg_train = Cost_sum / n_step
        # Cost_list_train.append(Cost_avg_train)
        P_tr_final = ApplyNetwork_sigmoid(X_tr, trained_net)
        Loss_avg_train = ComputeLoss_mlce(P_tr_final, Y_tr)
        Cost_avg_train = Loss_avg_train + lam*np.sum(trained_net['W']**2)

        Loss_list_train.append(Loss_avg_train)
        Cost_list_train.append(Cost_avg_train)
        # compute loss on validation data
        P_val = ApplyNetwork_sigmoid(X_val, trained_net)
        L_val = ComputeLoss_mlce(P_val, Y_val)
        Cost_val = L_val + lam*np.sum(trained_net['W']**2)
        Loss_list_val.append(L_val)
        Cost_list_val.append(Cost_val)
    
    return trained_net, Loss_list_train, Loss_list_val, Cost_list_train, Cost_list_val

def PlotGroundTruthProbHistogram(P, y, bins=20, title=None):
    y = np.asarray(y).reshape(-1).astype(int)
    n = P.shape[1]

    # Predicted class for each test example
    y_pred = np.argmax(P, axis=0)
    correct_mask = (y_pred == y)

    # Probability of the ground-truth class on test 
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
P = ApplyNetwork_sigmoid(X_tr[:], init_net)


# Step 5: compute the loss function
L = ComputeLoss_mlce(P, Y_tr)
print(f"Loss: {L}")


# Step 6: compute the accuracy
acc = ComputeAccuracy(P, y_tr)
print(f"Accuracy: {acc}")


# Step 7: evaluate the gradients
lam  = 0
grads = BackwardPass_mlce(X_tr, Y_tr, P, init_net, lam)
torch_grads = ComputeGradsWithTorch_mlce(X_tr, Y_tr, init_net, lam)

# snipped check
d_small = 10
n_small = 3
# lam = 0
small_net = {}
small_net['W'] =  .01*rng.standard_normal(size = (10, d_small))
small_net['b'] = np.zeros((10,1))

X_small = X_tr[0:d_small, 0:d_small]
Y_small = Y_tr[:, 0:d_small]
y_small = y_tr[0:d_small]

P = ApplyNetwork_sigmoid(X_small, small_net)
my_grads = BackwardPass_mlce(X_small, Y_small, P, small_net, lam)
tor_grads = ComputeGradsWithTorch_mlce(X_small, Y_small, small_net, lam)

## absolute aifference check
epi = 1e-6
ResultSame = True
for key in my_grads:
    diff = np.abs(my_grads[key] - tor_grads[key])
    if not np.all(diff < epi):
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
    if not np.all(rela_error < epi):
        print(f"{key} gradients differ!")
        ResultSame = False

if ResultSame:
    print("Two methods produce the same gradients")
else:
    print("Two methods produce different gradients")


# step 8:perform mini-batch gradient descent
n_batch = 100
eta = .01
n_epochs = 40
lam = 0.1
GDparams = {'n_batch': n_batch, 'eta': eta, 'epochs': n_epochs}
trained_net, Loss_list_train, Loss_list_val, Cost_list_train, Cost_list_val = MiniBatchGD(X_tr, Y_tr, GDparams, init_net, lam)

# combined loss and cost plot
# plt.plot(Cost_list_train, color = 'green')
# plt.plot(Loss_list_train, color = 'green', linestyle = '--')

# plt.plot(Cost_list_val, color = 'red')
# plt.plot(Loss_list_val, color = 'red', linestyle = '--')

# plt.xlim(left=0)
# plt.legend(['training Cost', 'training Loss', 'validation Cost', 'validation Loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('lambda = 0.1, eta = 0.01')
# # plt.savefig('Graph of Cost and Loss_04.png')
# plt.show()

# separated loss and cost plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(Cost_list_train, color='green', label='Training Cost')
plt.plot(Cost_list_val, color='red', label='Validation Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title(f'Cost (lambda={lam}, eta={eta})')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(Loss_list_train, color='green', label='Training Loss', linestyle='--')
plt.plot(Loss_list_val, color='red', label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss (lambda={lam}, eta={eta})')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
# plt.savefig('Separated_Loss_and_Cost.png')
plt.show()

print(f"First epoch Cost: {Cost_list_train[0]}")

# compute accuracy on test data
P_te = ApplyNetwork_sigmoid(X_te, trained_net)
acc_te = ComputeAccuracy(P_te, y_te)
print(f"Accuracy on test data:{acc_te*100:.2f}%")

# Histogram of ground-truth class probabilities (correct vs incorrect)
PlotGroundTruthProbHistogram(P_te, y_te, bins=20)

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

fig.suptitle('lambda = 0.1, eta = 0.01', fontsize=12, y=0.8) 
# plt.savefig('Visualization of weight matrix_04.png', bbox_inches='tight')
plt.show()

input("Press Enter to close...")


