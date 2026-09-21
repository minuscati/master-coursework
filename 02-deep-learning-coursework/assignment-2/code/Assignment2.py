# Coarse and fine search for lambda
#%%
import numpy as np
import pandas as pd
import pickle
import copy
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch
import torch
#%%
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
    
def NormalizeData(X, mean_X, std_X):
    X = (X-mean_X)/std_X
    return X

def InitializeNetwork(K, d, rng):
    init_net = {}
    init_net['W'] = (1/np.sqrt(d))*rng.standard_normal(size = (K,d))
    init_net['b'] = np.zeros((K,1))
    return init_net['W'], init_net['b']

#%%
def ApplyNetwork(X, net_params):
    fp_data = {}
    W1, b1 = net_params['W'][0], net_params['b'][0]
    W2, b2 = net_params['W'][1], net_params['b'][1]
    S1 = W1 @ X + b1 # m*n
    H1 = np.maximum(0, S1)
    S2 = W2 @ H1 + b2 # K*n
    P = np.exp(S2) / np.sum(np.exp(S2), axis=0)
    fp_data['H1'], fp_data['S1'],fp_data['P'] = H1,S1,P
    return P, fp_data

#%%
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

#%%
def BackwardPass(X, Y, fp_data, net_params, lam):
    P = fp_data['P']
    H1 = fp_data['H1']
    S1 = fp_data['S1']
    W1 = net_params['W'][0]
    W2 = net_params['W'][1]
    n = P.shape[1]
    K = Y.shape[0]
    m = W1.shape[0]
    grads = {'W': [None] * 2, 'b': [None] * 2}

    G1 = -(Y-P)
    grads['W'][1] = (1/n) * (G1 @ H1.T) + 2*lam * W2
    grads['b'][1] = (1/n) * np.sum(G1, axis=1).reshape(K,1)
    G2 = W2.T @ G1
    G2 = G2 * (S1 > 0)
    grads['W'][0] = (1/n) * (G2 @ X.T) + 2*lam * W1
    grads['b'][0] = (1/n) * np.sum(G2, axis=1).reshape(m,1)
    return grads


def SmallBatchGradientCheck(X, Y, y, d_small, n_small, m_hidden, lam, epi, seed):
    K = Y.shape[0]
    rng = np.random.default_rng(seed)
    small_net = {'W': [None] * 2, 'b': [None] * 2}
    small_net['W'][0] = (1 / np.sqrt(d_small)) * rng.standard_normal(size=(m_hidden, d_small))
    small_net['b'][0] = np.zeros((m_hidden, 1))
    small_net['W'][1] = (1 / np.sqrt(m_hidden)) * rng.standard_normal(size=(K, m_hidden))
    small_net['b'][1] = np.zeros((K, 1))

    X_small = X[0:d_small, 0:n_small]
    Y_small = Y[:, 0:n_small]
    y_small = np.asarray(y[0:n_small], dtype=np.int64)

    P_small, fp_data = ApplyNetwork(X_small, small_net)
    my_grads = BackwardPass(X_small, Y_small, fp_data, small_net, lam)
    tor_grads = ComputeGradsWithTorch(X_small, y_small, small_net, lam)

    same_gradients = True
    for key in ('W', 'b'):
        for i in range(len(my_grads[key])):
            mg = my_grads[key][i]
            tg = tor_grads[key][i]
            denom = np.maximum(epi, np.abs(tg) + np.abs(mg))
            rela_error = np.abs(tg - mg) / denom
            max_err = np.max(rela_error)
            print(f"Relative error of {key}[{i}]: {max_err}")
            if not np.all(rela_error < epi):
                print(f"{key}[{i}] gradients differ!")
                same_gradients = False

    print("Two methods produce the same gradients" if same_gradients else "Two methods produce different gradients")
    return same_gradients


#%%
def MiniBatchGD(X_tr, Y_tr, y_tr, GDparams, net_params, lam, X_val, y_val):
    n = X_tr.shape[1]
    n_batch = GDparams['n_batch']
    n_step = n // n_batch

    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    if 'eta' in GDparams:
        n_epochs = GDparams['epochs']
        eta_const = float(GDparams['eta'])
        use_cyclic = False
        record_every = n_step
    elif 'eta_min' in GDparams:
        eta_min = GDparams['eta_min']
        eta_max = GDparams['eta_max']
        ns = GDparams['ns']
        n_epochs = GDparams['n_epochs']
        use_cyclic = True

        if 'record_interval' in GDparams:
            record_every = max(1, (2*ns) // GDparams['record_interval'])
        else:
            record_every = n_step
    else:
        raise ValueError(
            "GDparams must include either ('eta', 'epochs') or ('eta_min', 'eta_max', 'ns', 'n_epochs')"
        )

    global_step = 0
    trained_net = copy.deepcopy(net_params)
    Loss_list_train, Cost_list_train, Accuracy_list_train = [], [], []
    Loss_list_val, Cost_list_val, Accuracy_list_val = [], [], []
    epoch_list = []  
    cumulative_batch_at_record = []

    def record_metrics(epoch_idx, cum_batches):
        # Full training / val sets (column order of X_tr does not change loss, only labels must match)
        P_tr, _ = ApplyNetwork(X_tr, trained_net)
        L_tr = ComputeLoss(P_tr, y_tr)
        Cost_tr = L_tr +  lam * sum(np.sum(w ** 2) for w in trained_net['W'])
        Acc_tr = ComputeAccuracy(P_tr, y_tr)

        P_val, _ = ApplyNetwork(X_val, trained_net)
        L_val = ComputeLoss(P_val, y_val)
        Cost_val = L_val +  lam * sum(np.sum(w ** 2) for w in trained_net['W'])
        Acc_val = ComputeAccuracy(P_val, y_val)

        epoch_list.append(epoch_idx)
        cumulative_batch_at_record.append(cum_batches)
        Loss_list_train.append(L_tr)
        Cost_list_train.append(Cost_tr)
        Accuracy_list_train.append(Acc_tr)
        Loss_list_val.append(L_val)
        Cost_list_val.append(Cost_val)
        Accuracy_list_val.append(Acc_val)


    record_metrics(0, global_step)
    # epoch loop
    for epoch in range(n_epochs):
        # shuffle training data 
        rng = np.random.default_rng(epoch)  
        perm = rng.permutation(n)
        X_tr = X_tr[:, perm]
        Y_tr = Y_tr[:, perm]
        y_tr = y_tr[perm]

        # batch loop
        for j in range(n_step):
            if use_cyclic:
                cycle = np.floor(1 + global_step / (2 * ns))
                x = np.abs(global_step / ns - 2 * cycle + 1)
                curr_eta = eta_min + (eta_max - eta_min) * np.maximum(0, (1 - x))
            else:
                curr_eta = eta_const
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            inds = range(j_start, j_end)
            Xbatch = X_tr[:,inds]
            Ybatch = Y_tr[:,inds]
            ybatch = np.argmax(Ybatch, axis=0)
            # forward
            Pbatch, fp_batch = ApplyNetwork(Xbatch, trained_net)
            Lbatch = ComputeLoss(Pbatch, ybatch)
            Cost = Lbatch + lam * sum(np.sum(w ** 2) for w in trained_net['W'])
            # backward
            grads = BackwardPass(Xbatch, Ybatch, fp_batch, trained_net, lam)
            # update
            for l in range(len(trained_net['W'])):
                trained_net['W'][l] -=  curr_eta * grads['W'][l]
                trained_net['b'][l] -=  curr_eta * grads['b'][l]

            global_step += 1
            if global_step % record_every == 0:
                record_metrics(epoch + 1, global_step)

    metrices = {
        'epoch': epoch_list,
        'cumulative_batches': cumulative_batch_at_record,
        'loss_tr': Loss_list_train,
        'cost_tr': Cost_list_train,
        'acc_tr': Accuracy_list_train,
        'loss_val': Loss_list_val,
        'cost_val': Cost_list_val,
        'acc_val': Accuracy_list_val,
    }
    
    return trained_net, metrices


#%%
def EvaluateLambda(lam_value, GDparams, X_tr, Y_tr, y_tr, X_val, y_val):
    [d, n] = X_tr.shape
    K = Y_tr.shape[0]
    m = 50
    net_params = {}
    net_params['W'] = [None] * 2
    net_params['b'] = [None] * 2
    
    rng = np.random.default_rng()
    net_params['W'][0], net_params['b'][0] = InitializeNetwork(m, d, rng)
    net_params['W'][1], net_params['b'][1] = InitializeNetwork(K, m, rng)
    
    trained_net, metrics = MiniBatchGD(
        X_tr, Y_tr, y_tr, GDparams, net_params, lam_value, X_val, y_val
    )
    
    best_val_acc = max(metrics['acc_val'])
    
    return best_val_acc, trained_net, metrics


#%%
def ComputeGradsWithTorch(X, y, network_params, lam):
    
    Xt = torch.from_numpy(X)
    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    S1 = W[0] @ Xt + b[0]
    H = apply_relu(S1)
    scores= W[1] @ H + b[1]
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n 

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    n = X.shape[1]
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    cost = loss + 2*lam * (torch.sum(W[0]**2) + torch.sum(W[1]**2))

    # compute the backward pass relative to the loss and the named parameters 
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads

#%% MAIN
cifar_dir = 'D:\\APPDATA\\Coding\\DL_course\\Datasets\\cifar-10-batches-py'
X_list = []
Y_list = []
y_list= []

for i in range(1,6):
    X_i, Y_i,y_i = LoadBatch(cifar_dir + '\\data_batch_' +str(i))
    X_list.append(X_i)
    Y_list.append(Y_i)
    y_list.append(y_i)

X_all = np.concatenate(X_list, axis=1)
Y_all= np.concatenate(Y_list, axis=1)
y_all = np.concatenate(y_list, axis=0)

size_val = 5000
X_val = X_all[:, -size_val:]
Y_val = Y_all[:, -size_val:]
y_val = y_all[-size_val:]

X_tr = X_all[:, :-size_val]
Y_tr = Y_all[:, :-size_val]
y_tr = y_all[:-size_val]

X_te, Y_te, y_te = LoadBatch(cifar_dir + '\\test_batch')


#%% test
print(X_tr.shape)
print(Y_tr.shape)
print(len(y_tr))

#%%
[d, n] = X_tr.shape
K = Y_tr.shape[0]
mean_X = np.mean(X_tr, axis=1).reshape(d,1)
std_X = np.std(X_tr, axis=1).reshape(d,1)
X_tr = NormalizeData(X_tr, mean_X, std_X)
X_val = NormalizeData(X_val, mean_X, std_X)
X_te = NormalizeData(X_te, mean_X, std_X)

#%%
L = 2 # number of layers
m = 50 # number of nodes in the hidden layer
net_params = {}
net_params['W'] = [None] * L
net_params['b'] = [None] * L 

seed = 42
rng = np.random.default_rng(seed)

net_params['W'][0], net_params['b'][0] = InitializeNetwork(m, d, rng)
net_params['W'][1], net_params['b'][1] = InitializeNetwork(K, m, rng)


#%%
lam = 0
_, fp_data = ApplyNetwork(X_tr, net_params)
my_grads = BackwardPass(X_tr, Y_tr, fp_data, net_params, lam)
torch_grads = ComputeGradsWithTorch(X_tr, y_tr, net_params, lam)
#%% small batch gradient check
_ = SmallBatchGradientCheck(X_tr, Y_tr, y_tr, d_small=5, n_small=3, m_hidden=6, lam=0, epi=1e-6, seed=42)

#%% overfit check: 100 examples, lam=0, ~200 epochs (gradient + MiniBatchGD OK if loss → very low)
n_overfit = 100
X_sub = X_tr[:, :n_overfit]
Y_sub = Y_tr[:, :n_overfit]
y_sub = y_tr[:n_overfit]
lam_overfit = 0
n_batch = 100  # one full pass over the 100 examples per epoch; use e.g. 10 if you want more steps per epoch
eta = 0.01
n_epochs = 200
GDparams = {'n_batch': n_batch, 'eta': eta, 'epochs': n_epochs, 'record_interval': 10}
trained_net, metrices = MiniBatchGD(
    X_sub, Y_sub, y_sub, GDparams, net_params, lam_overfit, X_val, y_val
)
print(f"Overfit: final avg batch train loss (last epoch) ≈ {metrices['loss_tr'][-1]:.6f}")

#%%
plt.figure(figsize=(8, 4.5))
plt.plot(metrices['epoch'], metrices['loss_tr'], label='train loss (full set)')
# plt.plot(metrices['epoch'], metrices['loss_val'], label='validation loss')
plt.xlabel('epoch (0 = before training)')
plt.ylabel('loss')
plt.title('Loss vs epoch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# input("Press Enter to close...")
# %%
# cyclical learning 
eta_min = 1e-5
eta_max = 1e-1
# ns = 500
ns = 800
n_batch = 100
n=X_tr.shape[1]
# lam = 0.01
n_step = n // n_batch
n_cycles = 2
n_epochs = int(np.ceil((n_cycles * 2 * ns) /n_step))

GDparams = {'n_batch': n_batch, 'eta_min': eta_min, 'eta_max': eta_max, 'ns': ns, 'n_epochs': n_epochs, 'record_interval': 9}
trained_net, metrices = MiniBatchGD(X_tr, Y_tr, y_tr, GDparams, net_params, lam, X_val, y_val)
step_ax = np.array(metrices['cumulative_batches'])
loss_tr, cost_tr, acc_tr = metrices['loss_tr'], metrices['cost_tr'], metrices['acc_tr']
loss_val, cost_val, acc_val = metrices['loss_val'], metrices['cost_val'], metrices['acc_val']

#%% plot cost (Cost = Loss + lam * sum(||W||^2); with small lam, cost ≈ loss + small offset)
plt.figure(figsize=(10, 6))
plt.plot(step_ax, cost_tr, color = 'green', label = 'training')
plt.plot(step_ax, cost_val, color = 'red', label = 'validation')
plt.xlabel('update step')
plt.ylabel('cost')
plt.ylim(0,4)
plt.xlim(left=0)
plt.legend()
plt.show()

#%% plot loss
plt.figure(figsize=(10, 6))
plt.plot(step_ax, loss_tr, color = 'green', label = 'training')
plt.plot(step_ax, loss_val, color = 'red', label = 'validation')
plt.xlabel('update step')
plt.ylabel('loss')
plt.ylim(0,3)
plt.xlim(left=0)
plt.legend()
plt.show()
# %% plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(step_ax, acc_tr, color = 'green', label = 'training')
plt.plot(step_ax, acc_val, color = 'red', label = 'validation')
plt.xlabel('update step')
plt.ylabel('accuracy')
plt.xlim(left=0)
plt.legend()
plt.show()

# test accuracy
P_te, _ = ApplyNetwork(X_te, trained_net)
Acc_te = ComputeAccuracy(P_te, y_te)
print(f"Test accuracy: {Acc_te:.6f}")
#%% Ex4
# coarse search
eta_min = 1e-5
eta_max = 1e-1
n_batch = 100
n=X_tr.shape[1]
ns = int(2 * np.floor(n / n_batch))
n_step = n // n_batch
n_cycles = 2
n_epochs = int(np.ceil((n_cycles * 2 * ns) /n_step))
GDparams = {'n_batch': n_batch, 'eta_min': eta_min, 'eta_max': eta_max, 'ns': ns, 'n_epochs': n_epochs, 'record_interval': 10}

l_min, l_max = -5, -1
search_time = 8
lambda_list = np.logspace(l_min, l_max, search_time)
search_results = []

for lmbda in lambda_list:
    best_acc, _, _ = EvaluateLambda(lmbda, GDparams, X_tr, Y_tr, y_tr, X_val, y_val)
    print(f"Lambda: {lmbda:.6f} | Best Validation Accuracy: {best_acc:.4f}")
    search_results.append((lmbda, best_acc))

search_results.sort(key=lambda x: x[1], reverse=True)
print("Top 3 Lambdas:", search_results[:3])

output_filename = "coarse_search_results.txt"
with open(output_filename, "w") as f:
    f.write("Lambda\tBest_Val_Accuracy\n")
    for lmbda, acc in search_results:
        f.write(f"{lmbda:.6e}\t{acc:.4f}\n")
print(f"Search results saved to {output_filename}")

#%%
# fine search
eta_min = 1e-5
eta_max = 1e-1
n_batch = 100
n=X_tr.shape[1]
ns = int(2 * np.floor(n / n_batch))
n_step = n // n_batch
n_cycles = 5
n_epochs = int(np.ceil((n_cycles * 2 * ns) /n_step))
GDparams = {'n_batch': n_batch, 'eta_min': eta_min, 'eta_max': eta_max, 'ns': ns, 'n_epochs': n_epochs, 'record_interval': 10}

l_min, l_max = -5.5, -3.0
search_time = 10
lambda_list = np.logspace(l_min, l_max, search_time)
search_results = []

for lmbda in lambda_list:
    best_acc, _, _ = EvaluateLambda(lmbda, GDparams, X_tr, Y_tr, y_tr, X_val, y_val)
    print(f"Lambda: {lmbda:.6f} | Best Validation Accuracy: {best_acc:.4f}")
    search_results.append((lmbda, best_acc))

search_results.sort(key=lambda x: x[1], reverse=True)
print("Top 3 Lambdas:", search_results[:3])

output_filename = "fine_search_results.txt"
with open(output_filename, "w") as f:
    f.write("Lambda\tBest_Val_Accuracy\n")
    for lmbda, acc in search_results:
        f.write(f"{lmbda:.6e}\t{acc:.4f}\n")
print(f"Search results saved to {output_filename}")

#%% best parameter
# best_lambda = search_results[0][0]
size_final_val = 1000
X_final_val = X_all[:, -size_final_val:]
Y_final_val = Y_all[:, -size_final_val:]
y_final_val = y_all[-size_final_val:]

X_final_tr = X_all[:, :-size_final_val]
Y_final_tr = Y_all[:, :-size_final_val]
y_final_tr = y_all[:-size_final_val]

[d, n] = X_final_tr.shape
K = Y_final_tr.shape[0]
mean_X = np.mean(X_final_tr, axis=1).reshape(d,1)
std_X = np.std(X_final_tr, axis=1).reshape(d,1)
X_final_tr = NormalizeData(X_final_tr, mean_X, std_X)
X_final_val = NormalizeData(X_final_val, mean_X, std_X)

L = 2 # number of layers
m = 50 # number of nodes in the hidden layer
net_params = {}
net_params['W'] = [None] * L
net_params['b'] = [None] * L 

seed = 42
rng = np.random.default_rng(seed)

net_params['W'][0], net_params['b'][0] = InitializeNetwork(m, d, rng)
net_params['W'][1], net_params['b'][1] = InitializeNetwork(K, m, rng)

#%% best parameter training
best_lam = 0.000527
n_batch = 100 
eta_min = 1e-5
eta_max = 1e-1
n=X_final_tr.shape[1]
ns = int(2 * np.floor(n / n_batch))
n_step = n // n_batch
n_cycles = 3
n_epochs = int(np.ceil((n_cycles * 2 * ns) /n_step))
GDparams = {'n_batch': n_batch, 'eta_min': eta_min, 'eta_max': eta_max, 'ns': ns, 'n_epochs': n_epochs, 'record_interval': 10}
trained_net, metrices = MiniBatchGD(
    X_final_tr, Y_final_tr, y_final_tr, GDparams, net_params, best_lam, X_final_val, y_final_val
)

#%% plot loss
step_ax = np.array(metrices['cumulative_batches'])
loss_tr, cost_tr, acc_tr = metrices['loss_tr'], metrices['cost_tr'], metrices['acc_tr']
loss_val, cost_val, acc_val = metrices['loss_val'], metrices['cost_val'], metrices['acc_val']

plt.figure(figsize=(10, 6))
plt.plot(step_ax, loss_tr, color = 'green', label = 'training')
plt.plot(step_ax, loss_val, color = 'red', label = 'validation')
plt.xlabel('update step')
plt.ylabel('loss')
plt.xlim(left=0)
plt.title('Training and Validation loss under best parameters')
plt.legend()
plt.show()

# test accuracy
P_te, _ = ApplyNetwork(X_te, trained_net)
Acc_te = ComputeAccuracy(P_te, y_te)
print(f"Test accuracy under best parameters: {Acc_te:.6f}")
# 52.01%