# Improvement: data augmentation 52.47%
#%%
import numpy as np
import pandas as pd
import pickle
import copy
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch
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
    """Shift a single CIFAR image by (tx, ty) with zero padding."""
    if abs(tx) >= 32 or abs(ty) >= 32:
        return np.zeros_like(xx)

    img = xx.reshape(3, 32, 32)
    tx_use, ty_use = tx, ty

    if tx_use < 0:
        img = img[:, :, ::-1]
        tx_use = -tx_use
    if ty_use < 0:
        img = img[:, ::-1, :]
        ty_use = -ty_use

    xx_work = img.reshape(3072, 1)
    xx_shifted = np.zeros_like(xx_work)

    aa = np.arange(32).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32 - tx_use))
    bb1 = np.arange(tx_use, 32, 1).reshape((32 - tx_use, 1))
    bb2 = np.arange(0, 32 - tx_use, 1).reshape((32 - tx_use, 1))
    ind_fill = vv.reshape((32 * (32 - tx_use), 1)) + np.tile(bb1, (32, 1))
    ii = np.transpose(np.nonzero(ind_fill > ty_use * 32 + 1))
    ind_fill = ind_fill[ii[0, 0]:]
    ind_xx = vv.reshape((32 * (32 - tx_use), 1)) + np.tile(bb2, (32, 1))
    ii = np.transpose(np.nonzero(ind_xx < 1024 - ty_use * 32))
    ind_xx = ind_xx[0:ii[-1, 0] + 1]

    # Safety alignment: for some (tx, ty), the two masks can differ by a few
    # entries; keep equal length before channel stacking.
    n_match = min(ind_fill.shape[0], ind_xx.shape[0])
    ind_fill = ind_fill[:n_match]
    ind_xx = ind_xx[:n_match]

    inds_fill = np.vstack((ind_fill, 1024 + ind_fill))
    inds_fill = np.vstack((inds_fill, 2048 + ind_fill))
    inds_xx = np.vstack((ind_xx, 1024 + ind_xx))
    inds_xx = np.vstack((inds_xx, 2048 + ind_xx))

    # Apply indices to produce the shifted image.
    inds_fill = inds_fill.astype(np.int64).ravel()
    inds_xx = inds_xx.astype(np.int64).ravel()
    xx_shifted[inds_fill, 0] = xx_work[inds_xx, 0]

    img_shifted = xx_shifted.reshape(3, 32, 32)
    if ty < 0:
        img_shifted = img_shifted[:, ::-1, :]
    if tx < 0:
        img_shifted = img_shifted[:, :, ::-1]
    return img_shifted.reshape(3072, 1)


def VisualizeShiftAugmentation(X, sample_idx=0):
    """Visualize original image and several shifted versions."""
    shifts = [(-3, -3), (-3, 3), (0, 0), (3, -3), (3, 3)]
    fig, axes = plt.subplots(1, len(shifts) + 1, figsize=(3 * (len(shifts) + 1), 3))

    img0 = X[:, sample_idx:sample_idx + 1].reshape(3, 32, 32).transpose(1, 2, 0)
    axes[0].imshow(np.clip(img0, 0, 1))
    axes[0].set_title("original")
    axes[0].axis("off")

    for i, (tx, ty) in enumerate(shifts, start=1):
        shifted = DataAugmentation(X[:, sample_idx:sample_idx + 1], tx, ty)
        img_shift = shifted.reshape(3, 32, 32).transpose(1, 2, 0)
        axes[i].imshow(np.clip(img_shift, 0, 1))
        axes[i].set_title(f"tx={tx}, ty={ty}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def InitializeNetwork(K, d, rng):
    # rng = np.random.default_rng()
    # BitGen = type(rng.bit_generator)
    # seed = 42
    # rng.bit_generator.state = BitGen(seed).state
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
    S2_shifted = S2 - np.max(S2, axis=0, keepdims=True)
    P = np.exp(S2_shifted) / np.sum(np.exp(S2_shifted), axis=0, keepdims=True)
    fp_data['H1'], fp_data['S1'],fp_data['P'] = H1,S1,P
    return P, fp_data

#%%
def ComputeLoss(P, y):
    n = P.shape[1]
    py = P[y, range(n)]
    l_ce = -np.log(py + 1e-12)
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
    mean_X = np.mean(X_tr, axis=1, keepdims=True)
    std_X  = np.std(X_tr, axis=1, keepdims=True)
    X_val_norm = NormalizeData(X_val, mean_X, std_X)

    def record_metrics(epoch_idx, cum_batches):
        # Full training / val sets (column order of X_tr does not change loss, only labels must match)
        X_tr_norm = NormalizeData(X_tr, mean_X, std_X)
        P_tr, _ = ApplyNetwork(X_tr_norm, trained_net)
        L_tr = ComputeLoss(P_tr, y_tr)
        Cost_tr = L_tr +  lam * sum(np.sum(w ** 2) for w in trained_net['W'])
        Acc_tr = ComputeAccuracy(P_tr, y_tr)

        P_val, _ = ApplyNetwork(X_val_norm, trained_net)
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

        # Use a conservative shift probability to limit slowdown from scattered memory access.
        shift_prob = 0.2

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
            Xbatch = X_tr[:,inds].copy()
            Ybatch = Y_tr[:,inds]
            ybatch = np.argmax(Ybatch, axis=0)

            for i in range(Xbatch.shape[1]):
                if np.random.rand() < 0.5:
                    img = Xbatch[:, i].reshape(3, 32, 32)
                    img_flipped = img[:, :, ::-1]
                    Xbatch[:, i] = img_flipped.reshape(3072)

                if np.random.rand() < shift_prob:
                    tx = np.random.randint(-3, 4)
                    ty = np.random.randint(-3, 4)
                    Xbatch[:, i:i+1] = DataAugmentation(Xbatch[:, i:i+1], tx, ty)


            Xbatch_norm = NormalizeData(Xbatch, mean_X, std_X)
            
            # forward
            Pbatch, fp_batch = ApplyNetwork(Xbatch_norm, trained_net)
            Lbatch = ComputeLoss(Pbatch, ybatch)
            Cost = Lbatch + lam * sum(np.sum(w ** 2) for w in trained_net['W'])
            # backward
            grads = BackwardPass(Xbatch_norm, Ybatch, fp_batch, trained_net, lam)
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

X_te, Y_te, y_te = LoadBatch(cifar_dir + '\\test_batch')

# Visual sanity check for shift augmentation (before/after).
VisualizeShiftAugmentation(X_all, sample_idx=0)


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
mean_X = np.mean(X_final_tr, axis=1, keepdims=True)
std_X = np.std(X_final_tr, axis=1, keepdims=True)
X_te = NormalizeData(X_te, mean_X, std_X)

L = 2 # number of layers
m = 50
net_params = {}
net_params['W'] = [None] * L
net_params['b'] = [None] * L 

seed = 42
rng = np.random.default_rng(seed)

#%% small batch gradient check
_ = SmallBatchGradientCheck(X_final_tr, Y_final_tr, y_final_tr, d_small=5, n_small=3, m_hidden=6, lam=0, epi=1e-6, seed=42)

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


net_params['W'][0], net_params['b'][0] = InitializeNetwork(m, d, rng)
net_params['W'][1], net_params['b'][1] = InitializeNetwork(K, m, rng)

trained_net, metrices = MiniBatchGD(
    X_final_tr, Y_final_tr, y_final_tr, GDparams, net_params, best_lam, X_final_val, y_final_val
)

P_te, _ = ApplyNetwork(X_te, trained_net)
Acc_te = ComputeAccuracy(P_te, y_te)
print(f"Test accuracy under best parameters： {Acc_te:.6f}")
# 52.47%