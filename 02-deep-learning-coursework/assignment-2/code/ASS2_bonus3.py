# Implement Adam optimizer + early stop +data augmentation + m +step decay
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

def DataAugmentation(xx, tx, ty):
    xx_shifted = np.zeros_like(xx)

    aa = np.arange(32).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32 - abs(tx)))

    if tx >= 0:
        bb1 = np.arange(tx, 32).reshape((32 - tx, 1))
        bb2 = np.arange(0, 32 - tx).reshape((32 - tx, 1))
    else:
        tx = -tx
        bb1 = np.arange(0, 32 - tx).reshape((32 - tx, 1))
        bb2 = np.arange(tx, 32).reshape((32 - tx, 1))

    ind_fill = vv.reshape((32 * (32 - abs(tx)), 1)) + np.tile(bb1, (32, 1))
    ind_xx = vv.reshape((32 * (32 - abs(tx)), 1)) + np.tile(bb2, (32, 1))

    if ty > 0:
        valid = ind_fill >= ty * 32
        ind_fill = ind_fill[valid]
        ind_xx = ind_xx[valid]
    elif ty < 0:
        ty = -ty
        valid = ind_xx < (1024 - ty * 32)
        ind_fill = ind_fill[valid]
        ind_xx = ind_xx[valid]

    inds_fill = np.vstack((ind_fill, 1024 + ind_fill, 2048 + ind_fill))
    inds_xx = np.vstack((ind_xx, 1024 + ind_xx, 2048 + ind_xx))

    xx_shifted[inds_fill] = xx[inds_xx]
    return xx_shifted

def HorizontalFlipBatch(X_batch, flip_mask):
    """
    X_batch: d x n_batch (CIFAR-10 flattened as 3x32x32)
    flip_mask: boolean array of length n_batch
    """
    if not np.any(flip_mask):
        return
    imgs = X_batch[:, flip_mask].reshape(3, 32, 32, -1)
    imgs = imgs[:, :, ::-1, :]
    X_batch[:, flip_mask] = imgs.reshape(3072, -1)

def RandomTranslateBatch(X_batch, shift_mask, max_shift=3):
    """
    Apply random translation only on selected samples.
    This keeps augmentation behavior while reducing Python overhead.
    """
    if not np.any(shift_mask):
        return
    sel_inds = np.where(shift_mask)[0]
    tx_all = np.random.randint(-max_shift, max_shift + 1, size=sel_inds.shape[0])
    ty_all = np.random.randint(-max_shift, max_shift + 1, size=sel_inds.shape[0])
    for k, idx in enumerate(sel_inds):
        X_batch[:, idx:idx+1] = DataAugmentation(X_batch[:, idx:idx+1], int(tx_all[k]), int(ty_all[k]))

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
    # Numerically stable softmax and avoid computing exp twice.
    S2_shift = S2 - np.max(S2, axis=0, keepdims=True)
    exp_S2 = np.exp(S2_shift)
    P = exp_S2 / np.sum(exp_S2, axis=0, keepdims=True)
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
    n_epochs = GDparams['n_epochs']
    base_eta = GDparams['eta']
    curr_eta = base_eta
    step_decay_size = int(GDparams.get('step_decay_size', 10))
    step_decay_gamma = float(GDparams.get('step_decay_gamma', 0.5))
    record_every_epochs = max(1, int(GDparams.get('record_interval', 1)))

    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)


    beta1, beta2, epsilon = GDparams['beta1'], GDparams['beta2'], GDparams['epsilon']
    global_step = 0
    trained_net = copy.deepcopy(net_params)
    patience = GDparams['patience']
    best_val_acc = 0
    patience_counter = 0
    best_net = copy.deepcopy(trained_net)


    m = {'W': [np.zeros_like(w) for w in trained_net['W']], 
         'b': [np.zeros_like(b) for b in trained_net['b']]}

    v = {'W': [np.zeros_like(w) for w in trained_net['W']], 
         'b': [np.zeros_like(b) for b in trained_net['b']]}

    t = 0

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
        # Step decay learning-rate schedule
        curr_eta = base_eta * (step_decay_gamma ** (epoch // step_decay_size))

        # shuffle training data 
        rng = np.random.default_rng(epoch)  
        perm = rng.permutation(n)
        X_tr = X_tr[:, perm]
        Y_tr = Y_tr[:, perm]
        y_tr = y_tr[perm]

        # batch loop
        for j in range(n_step):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            inds = range(j_start, j_end)
            Xbatch = X_tr[:,inds].copy()
            Ybatch = Y_tr[:,inds]

            # Batch-wise augmentation to reduce per-sample Python overhead.
            n_curr = Xbatch.shape[1]
            flip_mask = np.random.rand(n_curr) < GDparams.get('flip_prob', 0.5)
            shift_mask = np.random.rand(n_curr) < GDparams.get('shift_prob', 0.5)
            HorizontalFlipBatch(Xbatch, flip_mask)
            RandomTranslateBatch(Xbatch, shift_mask, max_shift=int(GDparams.get('max_shift', 3)))
            # forward
            Pbatch, fp_batch = ApplyNetwork(Xbatch, trained_net)
            # backward
            grads = BackwardPass(Xbatch, Ybatch, fp_batch, trained_net, lam)
            # update
            t += 1
            for l in range(len(trained_net['W'])):
                for key in ('W', 'b'):
                    g_t = grads[key][l]
                    m[key][l] = beta1 * m[key][l] + (1 - beta1) * g_t
                    v[key][l] = beta2 * v[key][l] + (1 - beta2) * g_t**2
                    m_hat = m[key][l] / (1 - beta1**t)
                    v_hat = v[key][l] / (1 - beta2**t)
                    trained_net[key][l] -= curr_eta * m_hat / (np.sqrt(v_hat) + epsilon)


            global_step += 1
        # Record at epoch granularity to avoid very expensive full-dataset evaluations per batch.
        # Record at epoch granularity to avoid very expensive full-dataset evaluations per batch.
        if (epoch + 1) % record_every_epochs == 0:
            record_metrics(epoch + 1, global_step)
            
            current_val_acc = Accuracy_list_val[-1] 
            
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                patience_counter = 0 
                best_trained_net = copy.deepcopy(trained_net) 
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping! Model did not improve for {patience} epochs, stopping at epoch {epoch+1}.")
                break


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
    
    return best_trained_net, metrices


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

#%% ----------MAIN Function---------
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

#%% overfit check
n_overfit = 100
X_sub = X_tr[:, :n_overfit]
Y_sub = Y_tr[:, :n_overfit]
y_sub = y_tr[:n_overfit]
lam_overfit = 0
n_batch = 100  # one full pass over the 100 examples per epoch; use e.g. 10 if you want more steps per epoch
eta = 0.001
n_epochs = 200
GDparams = {
    'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs, 'record_interval': 10,
    'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'patience': 15,
    'step_decay_size': 10, 'step_decay_gamma': 0.5,
    'flip_prob': 0.5, 'shift_prob': 0.5, 'max_shift': 3
}
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
# %% Real training
n_batch = 100
n=X_tr.shape[1]
n_epochs = 50
eta = 0.001
GDparams = {
    'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs, 'record_interval': 10,
    'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'patience': 15,
    'step_decay_size': 10, 'step_decay_gamma': 0.5,
    'flip_prob': 0.5, 'shift_prob': 0.5, 'max_shift': 3
}
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
# plt.ylim(0,4)
plt.xlim(left=0)
plt.legend()
plt.show()

#%% plot loss
plt.figure(figsize=(10, 6))
plt.plot(step_ax, loss_tr, color = 'green', label = 'training')
plt.plot(step_ax, loss_val, color = 'red', label = 'validation')
plt.xlabel('update step')
plt.ylabel('loss')
# plt.ylim(0,3)
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
#% 46.68%

#%% Coarse and fine search for lambda
# answer:2.77 * 10^-4
#%% best parameter
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
m = 400 # number of nodes in the hidden layer
net_params = {}
net_params['W'] = [None] * L
net_params['b'] = [None] * L 

seed = 42
rng = np.random.default_rng(seed)

net_params['W'][0], net_params['b'][0] = InitializeNetwork(m, d, rng)
net_params['W'][1], net_params['b'][1] = InitializeNetwork(K, m, rng)

#%% best parameter training
best_lam = 2.77e-4
n_batch = 100 
eta = 0.001
n_epochs = 50
GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs, 'record_interval': 10, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'patience': 15}
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
# %52.28%