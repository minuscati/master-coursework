# Semi-extensive testing
#%%
import numpy as np
import pandas as pd
import pickle
import copy
import os
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch
#%%
def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding = 'bytes')

    X = data_dict[b'data'].astype(np.float32)/255.0 # n*d(10000*3072)
    X = X.transpose() # d*n(3072*10000)

    y = data_dict[b'labels'] # n*1

    # one-hot encoding of the labels
    Y = pd.get_dummies(y).values.astype(np.float32)
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


def PlotTrainingCurves(metrices, title_prefix="Training Curves", save_path=None):
    """Plot train/val loss and accuracy curves from MiniBatchGD outputs."""
    x = metrices['cumulative_batches']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, metrices['loss_tr'], label='train_loss')
    axes[0].plot(x, metrices['loss_val'], label='val_loss')
    axes[0].set_xlabel('cumulative batches')
    axes[0].set_ylabel('loss')
    axes[0].set_title(f"{title_prefix} - Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.3)

    axes[1].plot(x, metrices['acc_tr'], label='train_acc')
    axes[1].plot(x, metrices['acc_val'], label='val_acc')
    axes[1].set_xlabel('cumulative batches')
    axes[1].set_ylabel('accuracy')
    axes[1].set_title(f"{title_prefix} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves figure to: {save_path}")
    plt.show()

def InitializeNetwork(K, d, rng):
    # rng = np.random.default_rng()
    # BitGen = type(rng.bit_generator)
    # seed = 42
    # rng.bit_generator.state = BitGen(seed).state
    init_net = {}
    init_net['W'] = ((1/np.sqrt(d))*rng.standard_normal(size = (K,d))).astype(np.float32)
    init_net['b'] = np.zeros((K,1), dtype=np.float32)
    return init_net['W'], init_net['b']

#%%
def ApplyNetwork(X, net_params, p_dropout=1.0, is_training=True):
    fp_data = {}
    W1, b1 = net_params['W'][0], net_params['b'][0]
    W2, b2 = net_params['W'][1], net_params['b'][1]
    S1 = W1 @ X + b1 # m*n
    H1 = np.maximum(0, S1)
    if is_training and p_dropout < 1.0:
        # Inverted dropout: keep-prob = p_dropout, scale during training.
        U1 = (np.random.rand(*H1.shape) < p_dropout) / p_dropout
        H1 = H1 * U1
    else:
        U1 = np.ones_like(H1)
    S2 = W2 @ H1 + b2 # K*n
    S2_shifted = S2 - np.max(S2, axis=0, keepdims=True)
    P = np.exp(S2_shifted) / np.sum(np.exp(S2_shifted), axis=0, keepdims=True)
    fp_data['H1'], fp_data['S1'], fp_data['P'], fp_data['U1'] = H1, S1, P, U1
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
    U1 = fp_data['U1']
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
    G2 = G2 * U1
    G2 = G2 * (S1 > 0)
    grads['W'][0] = (1/n) * (G2 @ X.T) + 2*lam * W1
    grads['b'][0] = (1/n) * np.sum(G2, axis=1).reshape(m,1)
    return grads


def SmallBatchGradientCheck(X, Y, y, d_small, n_small, m_hidden, lam, epi, seed):
    K = Y.shape[0]
    rng = np.random.default_rng(seed)
    small_net = {'W': [None] * 2, 'b': [None] * 2}
    small_net['W'][0] = ((1 / np.sqrt(d_small)) * rng.standard_normal(size=(m_hidden, d_small))).astype(np.float32)
    small_net['b'][0] = np.zeros((m_hidden, 1), dtype=np.float32)
    small_net['W'][1] = ((1 / np.sqrt(m_hidden)) * rng.standard_normal(size=(K, m_hidden))).astype(np.float32)
    small_net['b'][1] = np.zeros((K, 1), dtype=np.float32)

    X_small = X[0:d_small, 0:n_small]
    Y_small = Y[:, 0:n_small]
    y_small = np.asarray(y[0:n_small], dtype=np.int64)

    P_small, fp_data = ApplyNetwork(X_small, small_net, 1.0, is_training=False)
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
def BuildAugmentationBank(X_train_raw, t_range=3):
    aug_bank = {}
    print("Start building augmentation bank...")
    
    for tx in range(-t_range, t_range + 1):
        for ty in range(-t_range, t_range + 1):
            X_shifted = np.zeros_like(X_train_raw)
            
            for i in range(X_train_raw.shape[1]):
                X_shifted[:, i:i+1] = DataAugmentation(X_train_raw[:, i:i+1], tx, ty)
                
            aug_bank[(tx, ty)] = X_shifted
            print(f"Finished shift combination: ({tx}, {ty})")
            
    return aug_bank

#%%
def MiniBatchGD(X_tr, Y_tr, y_tr, GDparams, net_params, lam, X_val, y_val, aug_bank):
    n = X_tr.shape[1]
    n_batch = GDparams['n_batch']
    n_step = n // n_batch
    p_dropout = float(GDparams.get('p_dropout', 1.0))
    shift_prob = float(GDparams.get('shift_prob', 0.05))

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
    X_val_norm = NormalizeData(X_val, mean_X, std_X).astype(np.float32)
    X_tr_norm = NormalizeData(X_tr, mean_X, std_X).astype(np.float32)

    def record_metrics(epoch_idx, cum_batches):
        # Full training / val sets (column order of X_tr does not change loss, only labels must match)
        P_tr, _ = ApplyNetwork(X_tr_norm, trained_net, p_dropout, is_training=False)
        L_tr = ComputeLoss(P_tr, y_tr)
        Cost_tr = L_tr +  lam * sum(np.sum(w ** 2) for w in trained_net['W'])
        Acc_tr = ComputeAccuracy(P_tr, y_tr)

        P_val, _ = ApplyNetwork(X_val_norm, trained_net, p_dropout, is_training=False)
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
        X_tr_norm = X_tr_norm[:, perm]

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
            Xbatch = X_tr[:, inds].copy()
            Ybatch = Y_tr[:,inds]
            ybatch = np.argmax(Ybatch, axis=0)
            batch_size = Xbatch.shape[1]
            flip_mask = rng.random(batch_size) < 0.5
            
            shift_mask = rng.random(batch_size) < shift_prob
            shift_inds = np.where(shift_mask)[0]
            if shift_inds.size > 0:
                tx_choices = rng.integers(-3, 4, size=shift_inds.size)
                ty_choices = rng.integers(-3, 4, size=shift_inds.size)
                for local_idx, tx, ty in zip(shift_inds, tx_choices, ty_choices):
                    global_idx = inds[local_idx]
                    Xbatch[:, local_idx:local_idx+1] = aug_bank[(int(tx), int(ty))][:, global_idx:global_idx+1]

            if rng.random() < 0.5:
                Xbatch = Xbatch.reshape(3, 32, 32, -1)[:, :, ::-1, :].reshape(3072, -1)
            
            if np.any(flip_mask):
                Xbatch_img = Xbatch.T.reshape(batch_size, 3, 32, 32)
                Xbatch_img[flip_mask] = Xbatch_img[flip_mask][:, :, :, ::-1]
                Xbatch = Xbatch_img.reshape(batch_size, 3072).T

            Xbatch_norm = NormalizeData(Xbatch, mean_X, std_X).astype(np.float32)
            
            # forward
            Pbatch, fp_batch = ApplyNetwork(Xbatch_norm, trained_net, p_dropout, is_training=True)
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
VISUALIZE_AUG = False
if VISUALIZE_AUG:
    VisualizeShiftAugmentation(X_all, sample_idx=0)


if os.path.exists("aug_bank.npy"):
    aug_bank = np.load("aug_bank.npy", allow_pickle=True).item()
else:
    aug_bank = BuildAugmentationBank(X_all)
    np.save("aug_bank.npy", aug_bank) # 保存到硬盘
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
m = 1024
net_params = {}
net_params['W'] = [None] * L
net_params['b'] = [None] * L 

seed = 42
rng = np.random.default_rng(seed)

#%% small batch gradient check
_ = SmallBatchGradientCheck(X_final_tr, Y_final_tr, y_final_tr, d_small=5, n_small=3, m_hidden=6, lam=0, epi=1e-6, seed=42)

#%% random search framework (coarse random + local fine random)
def SampleLogUniform(rng, low, high):
    return 10 ** rng.uniform(np.log10(low), np.log10(high))


def BuildCyclicGDParams(n_train, n_batch, eta_max, p_dropout, n_cycles, eta_min=1e-5, shift_prob=0.05):
    ns = int(2 * np.floor(n_train / n_batch))
    n_step = n_train // n_batch
    n_epochs = int(np.ceil((n_cycles * 2 * ns) / n_step))
    return {
        'n_batch': n_batch,
        'eta_min': eta_min,
        'eta_max': float(eta_max),
        'ns': ns,
        'n_epochs': n_epochs,
        'record_interval': 2,
        'p_dropout': float(p_dropout),
        'shift_prob': float(shift_prob)
    }


def RunOneTrial(trial_id, lam, eta_max, p_dropout, n_cycles, n_batch, base_seed):
    trial_rng = np.random.default_rng(base_seed + 1009 * trial_id)
    trial_net = {'W': [None] * L, 'b': [None] * L}
    trial_net['W'][0], trial_net['b'][0] = InitializeNetwork(m, d, trial_rng)
    trial_net['W'][1], trial_net['b'][1] = InitializeNetwork(K, m, trial_rng)
    trial_gd = BuildCyclicGDParams(X_final_tr.shape[1], n_batch, eta_max, p_dropout, n_cycles=n_cycles)

    trained_net, metrics = MiniBatchGD(
        X_final_tr, Y_final_tr, y_final_tr, trial_gd, trial_net, lam, X_final_val, y_final_val, aug_bank
    )
    best_val_acc = float(np.max(metrics['acc_val']))
    final_val_acc = float(metrics['acc_val'][-1])
    return {
        'trial_id': trial_id,
        'lambda': float(lam),
        'eta_max': float(eta_max),
        'p_dropout': float(p_dropout),
        'best_val_acc': best_val_acc,
        'final_val_acc': final_val_acc,
        'gdparams': trial_gd,
        'trained_net': trained_net
    }


search_rng = np.random.default_rng(2026)
n_batch = 100
P_DROPOUT_CANDIDATES = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)

# Stage 1: coarse random search
n_coarse = 30
n_cycles_coarse = 1
coarse_results = []
for t in range(n_coarse):
    lam_t = SampleLogUniform(search_rng, 1e-6, 1e-2)
    eta_max_t = SampleLogUniform(search_rng, 1e-2, 3e-1)
    p_dropout_t = float(search_rng.choice(P_DROPOUT_CANDIDATES))  # keep probability
    res_t = RunOneTrial(
        trial_id=t,
        lam=lam_t,
        eta_max=eta_max_t,
        p_dropout=p_dropout_t,
        n_cycles=n_cycles_coarse,
        n_batch=n_batch,
        base_seed=42
    )
    coarse_results.append(res_t)
    print(
        f"[Coarse {t+1:02d}/{n_coarse}] val_best={res_t['best_val_acc']:.4f}, "
        f"val_final={res_t['final_val_acc']:.4f}, "
        f"lam={lam_t:.2e}, eta_max={eta_max_t:.2e}, p_drop={p_dropout_t:.1f}"
    )

coarse_results = sorted(coarse_results, key=lambda x: x['best_val_acc'], reverse=True)
coarse_for_csv = []
for rank_idx, row in enumerate(coarse_results, start=1):
    coarse_for_csv.append({
        'rank': rank_idx,
        'trial_id': row['trial_id'],
        'lambda': row['lambda'],
        'eta_max': row['eta_max'],
        'p_dropout': row['p_dropout'],
        'best_val_acc': row['best_val_acc'],
        'final_val_acc': row['final_val_acc']
    })
df_coarse = pd.DataFrame(coarse_for_csv)
coarse_csv_path = "random_search_coarse_results.csv"
df_coarse.to_csv(coarse_csv_path, index=False)
print(f"Saved coarse random search results to: {coarse_csv_path}")

top_k = 5
top_candidates = coarse_results[:top_k]

# Stage 2: local fine random search around top candidates
n_fine_per_top = 6
n_cycles_fine = 2
fine_results = []
trial_counter = n_coarse
for base in top_candidates:
    for _ in range(n_fine_per_top):
        lam_f = np.clip(base['lambda'] * (10 ** search_rng.uniform(-0.5, 0.5)), 1e-7, 5e-2)
        eta_f = np.clip(base['eta_max'] * (10 ** search_rng.uniform(-0.4, 0.4)), 5e-3, 5e-1)
        p_f = np.clip(base['p_dropout'] + search_rng.uniform(-0.08, 0.08), 0.6, 1.0)
        res_f = RunOneTrial(
            trial_id=trial_counter,
            lam=lam_f,
            eta_max=eta_f,
            p_dropout=p_f,
            n_cycles=n_cycles_fine,
            n_batch=n_batch,
            base_seed=42
        )
        fine_results.append(res_f)
        print(
            f"[Fine {len(fine_results):02d}/{top_k*n_fine_per_top}] val_best={res_f['best_val_acc']:.4f}, "
            f"lam={lam_f:.2e}, eta_max={eta_f:.2e}, p_drop={p_f:.3f}"
        )
        trial_counter += 1

all_results = coarse_results + fine_results
all_results = sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True)
best_cfg = all_results[0]

# Save all search trials to CSV 
results_for_csv = []
for rank_idx, row in enumerate(all_results, start=1):
    results_for_csv.append({
        'rank': rank_idx,
        'trial_id': row['trial_id'],
        'lambda': row['lambda'],
        'eta_max': row['eta_max'],
        'p_dropout': row['p_dropout'],
        'best_val_acc': row['best_val_acc'],
        'final_val_acc': row['final_val_acc']
    })
df_search = pd.DataFrame(results_for_csv)
csv_path = "random_search_results.csv"
df_search.to_csv(csv_path, index=False)
print(f"Saved random search results to: {csv_path}")

print("\nBest hyperparameters from random search:")
print(
    f"lambda={best_cfg['lambda']:.6e}, eta_max={best_cfg['eta_max']:.6e}, "
    f"p_dropout={best_cfg['p_dropout']:.4f}, val_best={best_cfg['best_val_acc']:.4f}"
)

best_cfg_path = "best_hyperparams2.csv"
pd.DataFrame([{
    'lambda': best_cfg['lambda'],
    'eta_max': best_cfg['eta_max'],
    'p_dropout': best_cfg['p_dropout'],
    'best_val_acc': best_cfg['best_val_acc']
}]).to_csv(best_cfg_path, index=False)
print(f"Saved best hyperparameters to: {best_cfg_path}")


# Final run
best_cfg = pd.read_csv("best_hyperparams2.csv").iloc[0]
n_batch = 100
phase1_cycles = 6
phase2_cycles = 4
anneal_factor = 0.3

final_rng = np.random.default_rng(999)
final_net = {'W': [None] * L, 'b': [None] * L}
final_net['W'][0], final_net['b'][0] = InitializeNetwork(m, d, final_rng)
final_net['W'][1], final_net['b'][1] = InitializeNetwork(K, m, final_rng)

print("\n[Phase 1] regular cyclic training")
gd_phase1 = BuildCyclicGDParams(
    X_final_tr.shape[1], n_batch, best_cfg['eta_max'], best_cfg['p_dropout'],
    n_cycles=phase1_cycles
)
net_after_phase1, metrics_phase1 = MiniBatchGD(
    X_final_tr, Y_final_tr, y_final_tr, gd_phase1, final_net,
    best_cfg['lambda'], X_final_val, y_final_val, aug_bank
)

print("\n[Phase 2] annealing training")
eta_max_phase2 = float(best_cfg['eta_max']) * anneal_factor
eta_min_phase2 = float(gd_phase1['eta_min']) * anneal_factor
gd_phase2 = BuildCyclicGDParams(
    X_final_tr.shape[1], n_batch, eta_max_phase2, best_cfg['p_dropout'],
    n_cycles=phase2_cycles, eta_min=eta_min_phase2
)
trained_net, metrics_phase2 = MiniBatchGD(
    X_final_tr, Y_final_tr, y_final_tr, gd_phase2, net_after_phase1,
    best_cfg['lambda'], X_final_val, y_final_val, aug_bank
)


batch_offset = metrics_phase1['cumulative_batches'][-1]
epoch_offset = metrics_phase1['epoch'][-1]
metrices = {}
metrices['cumulative_batches'] = metrics_phase1['cumulative_batches'] + [
    x + batch_offset for x in metrics_phase2['cumulative_batches'][1:]
]
metrices['epoch'] = metrics_phase1['epoch'] + [
    x + epoch_offset for x in metrics_phase2['epoch'][1:]
]
for key in ('loss_tr', 'cost_tr', 'acc_tr', 'loss_val', 'cost_val', 'acc_val'):
    metrices[key] = metrics_phase1[key] + metrics_phase2[key][1:]

P_te, _ = ApplyNetwork(X_te, trained_net, best_cfg['p_dropout'], is_training=False)
Acc_te = ComputeAccuracy(P_te, y_te)
print(f"Test accuracy under searched best parameters: {Acc_te:.6f}")

PlotTrainingCurves(
    metrices,
    title_prefix="Final Long Run",
    save_path="final_long_run_curves.png"
)
