# ASS3_1.2
#%%
import numpy as np
import pandas as pd
import pickle
import copy
import os
import matplotlib.pyplot as plt
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

def MX_Generator(X, f, n_p=None):
    """Build MX with shape (n_p, 3*f*f, n) from X shaped (3072, n)."""
    d, n = X.shape
    if d != 32 * 32 * 3:
        raise ValueError(f"Expected d=3072, got d={d}")
    if 32 % f != 0:
        raise ValueError(f"f must divide 32, got f={f}")

    out_h = 32 // f
    out_w = 32 // f
    n_p = out_h * out_w
    patch_dim = 3 * f * f

    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    MX = np.zeros((n_p, patch_dim, n), dtype=X.dtype)

    for i in range(n):
        l = 0
        for r in range(out_h):
            r0 = r * f
            r1 = r0 + f
            for c in range(out_w):
                c0 = c * f
                c1 = c0 + f
                MX[l, :, i] = X_ims[r0:r1, c0:c1, :, i].reshape((patch_dim,), order='C')
                l += 1
    return MX


def get_or_create_mx_cache(X, f, cache_path):
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            MX = data['MX']
    else:
        MX = MX_Generator(X, f)
        np.savez(cache_path, MX=MX)
    return MX


def InitializeNetwork(*args, dtype=np.float64):
    if len(args) != 5:
        raise ValueError("InitializeNetwork expects 5 args.")

    f, nf, nh, K, rng = args
    """He initialize conv+fc parameters for the convnet."""
    patch_dim = 3 * f * f
    n_p = (32 // f) * (32 // f)
    conv_flat_dim = n_p * nf

    Fs_flat = rng.standard_normal(size=(patch_dim, nf)).astype(dtype) * np.sqrt(2.0 / patch_dim)
    b_conv = np.zeros((nf, 1), dtype=dtype)
    W1 = rng.standard_normal(size=(nh, conv_flat_dim)).astype(dtype) * np.sqrt(2.0 / conv_flat_dim)
    b1 = np.zeros((nh, 1), dtype=dtype)
    W2 = rng.standard_normal(size=(K, nh)).astype(dtype) * np.sqrt(2.0 / nh)
    b2 = np.zeros((K, 1), dtype=dtype)
    return W1, b1, W2, b2, b_conv, Fs_flat


#%%
def ComputeLoss(P, y):
    n = P.shape[1]
    py = P[y, range(n)]
    return np.mean(-np.log(py + 1e-12))


def cross_entropy_with_logits_targets(P, targets):
    return np.mean(-np.sum(targets * np.log(P + 1e-12), axis=0))


def ComputeAccuracy(P, y):
    n = P.shape[1]
    y_tr_pred = np.argmax(P, axis=0)
    return sum(y_tr_pred == y) / n


def ReLu(x):
    return np.maximum(0, x)


def softmax(S):
    S_shift = S - np.max(S, axis=0, keepdims=True)
    nominators = np.exp(S_shift)
    denominators = np.sum(nominators, axis=0, keepdims=True)
    return nominators / denominators


def label_smoothing(Y, epsilon=0.1):
    Y = np.asarray(Y, dtype=np.float64)
    K = Y.shape[0]
    if K <= 1:
        return Y.copy()
    fill = epsilon / (K - 1)
    return Y * (1.0 - epsilon - fill) + fill


def forward_pass(MX, Fs_flat, W1, b1, W2, b2, b_conv):
    n_p, _, n = MX.shape
    nf = Fs_flat.shape[1]
    H1 = np.einsum('ijn,jl->iln', MX, Fs_flat, optimize=True) + b_conv
    conv_flat = ReLu(H1.reshape((n_p * nf, n), order='C'))
    X1_pre = np.matmul(W1, conv_flat) + b1
    X1 = ReLu(X1_pre)
    S = np.matmul(W2, X1) + b2
    P = softmax(S)
    Mid_params = {'H1': H1, 'conv_flat': conv_flat, 'X1_pre': X1_pre, 'X1': X1, 'S': S}
    return P, Mid_params


def backward_pass(P, MX, Y_targets, Fs_Flat, W1, b1, W2, b2, b_conv, lam, Mid_params):
    _, conv_flat, X1_pre, X1, _ = Mid_params.values()
    n = Y_targets.shape[1]
    n_p = MX.shape[0]
    nf = Fs_Flat.shape[1]

    # softmax + CE: dL/dS = P - Y_targets (Y_targets one-hot or label-smoothed)
    G = P - Y_targets
    grad_W2 = (1 / n) * np.matmul(G, X1.T) + 2 * lam * W2
    grad_b2 = (1 / n) * np.sum(G, axis=1, keepdims=True)

    G = np.matmul(W2.T, G)
    G = G * (X1_pre > 0)

    grad_W1 = (1 / n) * np.matmul(G, conv_flat.T) + 2 * lam * W1
    grad_b1 = (1 / n) * np.sum(G, axis=1, keepdims=True)

    G_batch = np.matmul(W1.T, G)
    G_batch = G_batch * (conv_flat > 0)
    GG = G_batch.reshape((n_p, nf, n), order='C')
    grad_b_conv = (1 / n) * np.sum(GG, axis=(0, 2), keepdims=False).reshape((-1, 1))

    Mxt = np.transpose(MX, (1, 0, 2))
    grad_Fs_flat = (1 / n) * np.einsum('ijn,jln->il', Mxt, GG, optimize=True) + 2 * lam * Fs_Flat

    return {
        'grad_W2': grad_W2,
        'grad_b2': grad_b2,
        'grad_W1': grad_W1,
        'grad_b1': grad_b1,
        'grad_Fs_flat': grad_Fs_flat,
        'grad_b_conv': grad_b_conv,
    }


def compare_grad(name, g_np, g_th):
    g_th_np = g_th.detach().cpu().numpy()
    abs_err = np.max(np.abs(g_np - g_th_np))
    rel_err = abs_err / max(1e-12, np.max(np.abs(g_np)) + np.max(np.abs(g_th_np)))
    print(f"{name:12s} shape={g_np.shape}, abs_err={abs_err:.3e}, rel_err={rel_err:.3e}")


def torch_conv_loop_check(MX, Y, net_params, lam, label_smoothing_epsilon=None):
    Fs_flat = net_params['Fs_flat'].astype(np.float64)
    b_conv = net_params['b_conv'].astype(np.float64)
    W1 = net_params['W1'].astype(np.float64)
    b1 = net_params['b1'].astype(np.float64)
    W2 = net_params['W2'].astype(np.float64)
    b2 = net_params['b2'].astype(np.float64)
    MX_np = MX.astype(np.float64)
    Y_np = Y.astype(np.float64)

    Y_targets_np = (
        label_smoothing(Y_np, label_smoothing_epsilon)
        if label_smoothing_epsilon is not None and label_smoothing_epsilon > 0
        else Y_np
    )
    P_np, mid_np = forward_pass(MX_np, Fs_flat, W1, b1, W2, b2, b_conv)
    grads_np = backward_pass(P_np, MX_np, Y_targets_np, Fs_flat, W1, b1, W2, b2, b_conv, lam, mid_np)

    MX_t = torch.tensor(MX_np, dtype=torch.float64)
    Y_t = torch.tensor(Y_targets_np, dtype=torch.float64)
    Fs_t = torch.tensor(Fs_flat, dtype=torch.float64, requires_grad=True)
    b_conv_t = torch.tensor(b_conv, dtype=torch.float64, requires_grad=True)
    W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    n_p, _, n = MX_t.shape
    nf = Fs_t.shape[1]
    H1_t = torch.zeros((n_p, nf, n), dtype=torch.float64)
    for i in range(n):
        H1_t[:, :, i] = MX_t[:, :, i] @ Fs_t
    H1_t = H1_t + b_conv_t

    conv_flat_t = torch.relu(H1_t.reshape(n_p * nf, n))
    X1_pre_t = W1_t @ conv_flat_t + b1_t
    X1_t = torch.relu(X1_pre_t)
    S_t = W2_t @ X1_t + b2_t
    P_t = torch.softmax(S_t, dim=0)

    loss_ce = -torch.sum(Y_t * torch.log(P_t + 1e-12)) / n
    loss_reg = lam * (torch.sum(W1_t * W1_t) + torch.sum(W2_t * W2_t) + torch.sum(Fs_t * Fs_t))
    (loss_ce + loss_reg).backward()

    print("\n[Torch for-loop conv gradient check]")
    compare_grad('W2', grads_np['grad_W2'], W2_t.grad)
    compare_grad('b2', grads_np['grad_b2'], b2_t.grad)
    compare_grad('W1', grads_np['grad_W1'], W1_t.grad)
    compare_grad('b1', grads_np['grad_b1'], b1_t.grad)
    compare_grad('Fs_flat', grads_np['grad_Fs_flat'], Fs_t.grad)
    compare_grad('b_conv', grads_np['grad_b_conv'], b_conv_t.grad)


def MiniBatchGD(
    MX_tr, Y_tr, y_tr, GDparams, net_params, lam,
    MX_te=None, y_te=None, record_test_every=1, test_eval_size=10000,
    label_smoothing_epsilon=None,
):
    n = MX_tr.shape[2]
    n_batch = GDparams['n_batch']
    n_step = n // n_batch

    y_tr = np.asarray(y_tr, dtype=np.int64)
    track_test = (MX_te is not None) and (y_te is not None)
    if track_test:
        y_te = np.asarray(y_te, dtype=np.int64)
        te_n = MX_te.shape[2]
        m_te = min(test_eval_size, te_n)
        test_idx = np.arange(m_te)

    if 'eta' in GDparams:
        n_epochs = GDparams['epochs']
        eta_const = float(GDparams['eta'])
        use_cyclic = False
        record_every = n_step
    elif 'eta_min' in GDparams:
        eta_min = GDparams['eta_min']
        eta_max = GDparams['eta_max']
        ns = GDparams.get('ns', None)
        ns_list = GDparams.get('ns_list', None)
        n_epochs = GDparams['n_epochs']
        use_cyclic = True
        if ns_list is not None:
            total_cycle_steps = int(np.sum([2 * s for s in ns_list]))
            record_every = max(1, total_cycle_steps // GDparams.get('record_interval', n_step))
        else:
            record_every = max(1, (2 * ns) // GDparams.get('record_interval', n_step))
    else:
        raise ValueError("GDparams must include either ('eta','epochs') or ('eta_min','eta_max','ns','n_epochs').")

    global_step = 0
    trained_net = copy.deepcopy(net_params)
    ls_eps = GDparams.get('label_smoothing_epsilon', label_smoothing_epsilon)
    use_ls = ls_eps is not None and float(ls_eps) > 0
    ls_eps = float(ls_eps) if use_ls else 0.0
    Loss_list_train, Cost_list_train, Accuracy_list_train = [], [], []
    Loss_list_test, Cost_list_test, Accuracy_list_test = [], [], []
    epoch_list = []
    cumulative_batch_at_record = []

    def calc_cost(loss_value):
        return loss_value + lam * (
            np.sum(trained_net['W1'] ** 2) + np.sum(trained_net['W2'] ** 2) + np.sum(trained_net['Fs_flat'] ** 2)
        )

    def run_fp(MX_input):
        return forward_pass(
            MX_input,
            trained_net['Fs_flat'],
            trained_net['W1'],
            trained_net['b1'],
            trained_net['W2'],
            trained_net['b2'],
            trained_net['b_conv'],
        )

    def eta_cyclic(global_step):
        if ns_list is None:
            cycle = np.floor(1 + global_step / (2 * ns))
            x = np.abs(global_step / ns - 2 * cycle + 1)
            return eta_min + (eta_max - eta_min) * np.maximum(0, (1 - x))

        # variable cycle length: ns doubles each cycle according to ns_list
        t = int(global_step)
        prev_end = 0
        for ns_i in ns_list:
            cycle_len = 2 * ns_i
            end = prev_end + cycle_len
            if t < end:
                local_t = t - prev_end
                if local_t <= ns_i:
                    return eta_min + (eta_max - eta_min) * (local_t / ns_i)
                return eta_max - (eta_max - eta_min) * ((local_t - ns_i) / ns_i)
            prev_end = end

        # fallback if training runs beyond configured cycles: keep last cycle length
        ns_i = ns_list[-1]
        local_t = (t - prev_end) % (2 * ns_i)
        if local_t <= ns_i:
            return eta_min + (eta_max - eta_min) * (local_t / ns_i)
        return eta_max - (eta_max - eta_min) * ((local_t - ns_i) / ns_i)

    def record_metrics(epoch_idx, cum_batches):
        P_tr, _ = run_fp(MX_tr)
        if use_ls:
            Y_smooth_tr = label_smoothing(Y_tr, ls_eps)
            L_tr = cross_entropy_with_logits_targets(P_tr, Y_smooth_tr)
        else:
            L_tr = ComputeLoss(P_tr, y_tr)
        Cost_tr = calc_cost(L_tr)
        Acc_tr = ComputeAccuracy(P_tr, y_tr)

        if track_test and (len(epoch_list) % record_test_every == 0):
            P_te, _ = run_fp(MX_te[:, :, test_idx])
            L_te = ComputeLoss(P_te, y_te[test_idx])
            Cost_te = calc_cost(L_te)
            Acc_te = ComputeAccuracy(P_te, y_te[test_idx])
        else:
            L_te = np.nan
            Cost_te = np.nan
            Acc_te = np.nan

        epoch_list.append(epoch_idx)
        cumulative_batch_at_record.append(cum_batches)
        Loss_list_train.append(L_tr)
        Cost_list_train.append(Cost_tr)
        Accuracy_list_train.append(Acc_tr)
        Loss_list_test.append(L_te)
        Cost_list_test.append(Cost_te)
        Accuracy_list_test.append(Acc_te)

    record_metrics(0, global_step)
    for epoch in range(n_epochs):
        rng = np.random.default_rng(epoch)
        perm = rng.permutation(n)
        MX_tr = MX_tr[:, :, perm]
        Y_tr = Y_tr[:, perm]
        y_tr = y_tr[perm]

        for j in range(n_step):
            if use_cyclic:
                curr_eta = eta_cyclic(global_step)
            else:
                curr_eta = eta_const

            inds = range(j * n_batch, (j + 1) * n_batch)
            MXbatch = MX_tr[:, :, inds]
            Ybatch = Y_tr[:, inds]
            Ybatch_targets = label_smoothing(Ybatch, ls_eps) if use_ls else Ybatch
            Pbatch, fp_batch = run_fp(MXbatch)
            grads = backward_pass(
                Pbatch, MXbatch, Ybatch_targets,
                trained_net['Fs_flat'], trained_net['W1'], trained_net['b1'],
                trained_net['W2'], trained_net['b2'], trained_net['b_conv'],
                lam, fp_batch
            )

            trained_net['W2'] -= curr_eta * grads['grad_W2']
            trained_net['b2'] -= curr_eta * grads['grad_b2']
            trained_net['W1'] -= curr_eta * grads['grad_W1']
            trained_net['b1'] -= curr_eta * grads['grad_b1']
            trained_net['Fs_flat'] -= curr_eta * grads['grad_Fs_flat']
            trained_net['b_conv'] -= curr_eta * grads['grad_b_conv']

            global_step += 1
            if global_step % record_every == 0:
                record_metrics(epoch + 1, global_step)

    metrics = {
        'epoch': epoch_list,
        'cumulative_batches': cumulative_batch_at_record,
        'loss_tr': Loss_list_train,
        'cost_tr': Cost_list_train,
        'acc_tr': Accuracy_list_train,
        'loss_te': Loss_list_test,
        'cost_te': Cost_list_test,
        'acc_te': Accuracy_list_test,
    }
    return trained_net, metrics


def EvaluateLambda(lam_value, GDparams, MX_tr, Y_tr, y_tr, f, nf, nh):
    K = Y_tr.shape[0]
    rng = np.random.default_rng()
    W1, b1, W2, b2, b_conv, Fs_flat = InitializeNetwork(f, nf, nh, K, rng)
    net_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'b_conv': b_conv, 'Fs_flat': Fs_flat}
    trained_net, metrics = MiniBatchGD(MX_tr, Y_tr, y_tr, GDparams, net_params, lam_value)
    best_train_acc = max(metrics['acc_tr'])
    return best_train_acc, trained_net, metrics

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

size_val = 1000
X_tr = X_all[:, :-size_val]
Y_tr = Y_all[:, :-size_val]
y_tr = y_all[:-size_val]

X_te, Y_te, y_te = LoadBatch(cifar_dir + '\\test_batch')

# MX should be built from normalized inputs; defer cache creation until after normalization.

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
X_te = NormalizeData(X_te, mean_X, std_X)

mx_cache_dir = os.path.join(cifar_dir, 'mx_cache')
os.makedirs(mx_cache_dir, exist_ok=True)
f_conv = 4
MX_tr = get_or_create_mx_cache(X_tr, f_conv, os.path.join(mx_cache_dir, f'MX_tr_f{f_conv}.npz'))
MX_te = get_or_create_mx_cache(X_te, f_conv, os.path.join(mx_cache_dir, f'MX_te_f{f_conv}.npz'))

#%% Gradient check on small net
nf_small = 5
nh_small = 6
n = 10
rng = np.random.default_rng(42)
W1, b1, W2, b2, b_conv, Fs_flat = InitializeNetwork(f_conv, nf_small, nh_small, K, rng)
net_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'b_conv': b_conv, 'Fs_flat': Fs_flat}
torch_conv_loop_check(MX_tr[:, :, :n], Y_tr[:, :n], net_params, lam=1e-3, label_smoothing_epsilon=0.1)

#%% Main training
nh = 50
nf_conv = 10

architectures = [
    # {'name': 'arch2', 'f': 4, 'nf': 10, 'nh': 50},
    # {'name': 'arch3', 'f': 8, 'nf': 40, 'nh': 50},
    # {'name': 'arch5', 'f': 4, 'nf': 40, 'nh': 50},
    {'name': 'arch6', 'f': 4, 'nf': 40, 'nh': 300},
]

seed = 42
rng = np.random.default_rng(seed)
W1, b1, W2, b2, b_conv, Fs_flat = InitializeNetwork(f_conv, nf_conv, nh, K, rng)
net_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'b_conv': b_conv, 'Fs_flat': Fs_flat}

lam = 2.5e-3
torch_conv_loop_check(MX_tr[:, :, :20], Y_tr[:, :20], net_params, lam=lam, label_smoothing_epsilon=0.1)

# %% cyclical learning rate
n_cycles = 4
ns_list = [800, 1600, 3200]
eta_min = 1e-5
eta_max = 1e-1
n_batch = 100
n = MX_tr.shape[2]
n_step = n // n_batch


#%% Exercise 3: architecture sweep (final test accuracy + training time)
arch_names = []
arch_test_acc = []

for arch in architectures:
    f_arch = arch['f']
    nf_arch = arch['nf']
    nh_arch = arch['nh']

    MX_tr_arch = get_or_create_mx_cache(X_tr, f_arch, os.path.join(mx_cache_dir, f'MX_tr_f{f_arch}.npz'))
    MX_te_arch = get_or_create_mx_cache(X_te, f_arch, os.path.join(mx_cache_dir, f'MX_te_f{f_arch}.npz'))

    n_arch = MX_tr_arch.shape[2]
    n_step_arch = n_arch // n_batch
    n_epochs_arch = int(np.ceil(np.sum([2 * s for s in ns_list]) / n_step_arch))
    GDparams_arch = {
        'n_batch': n_batch,
        'eta_min': eta_min,
        'eta_max': eta_max,
        'ns_list': ns_list,
        'n_epochs': n_epochs_arch,
        'record_interval': 9,
        'label_smoothing_epsilon': 0.1,
    }

    rng_arch = np.random.default_rng(seed)
    W1_a, b1_a, W2_a, b2_a, b_conv_a, Fs_flat_a = InitializeNetwork(f_arch, nf_arch, nh_arch, K, rng_arch)
    net_arch = {'W1': W1_a, 'b1': b1_a, 'W2': W2_a, 'b2': b2_a, 'b_conv': b_conv_a, 'Fs_flat': Fs_flat_a}

    # Training time only: optimizer loop start -> optimizer loop end.
    trained_arch, metrices = MiniBatchGD(
        MX_tr_arch, Y_tr, y_tr, GDparams_arch, net_arch, lam,
        MX_te=MX_te_arch, y_te=y_te, record_test_every=1, test_eval_size=10000
    )

    P_te_arch, _ = forward_pass(
        MX_te_arch,
        trained_arch['Fs_flat'],
        trained_arch['W1'],
        trained_arch['b1'],
        trained_arch['W2'],
        trained_arch['b2'],
        trained_arch['b_conv'],
    )
    acc_arch = ComputeAccuracy(P_te_arch, y_te)
#%%
    arch_names.append(f"f={f_arch}, nf={nf_arch}, nh={nh_arch}")
    arch_test_acc.append(acc_arch)

    step_ax = np.array(metrices['cumulative_batches'])
    loss_tr = np.array(metrices['loss_tr'])
    loss_te = np.array(metrices['loss_te'])
    mask_te = ~np.isnan(loss_te)

    plt.figure(figsize=(10, 6))
    plt.plot(step_ax, loss_tr, color='blue', label='training loss')
    plt.plot(step_ax[mask_te], loss_te[mask_te], color='orange', label='test loss')
    plt.xlabel('update step')
    plt.ylabel('loss')
    plt.title(f"{arch['name']} (f={f_arch}, nf={nf_arch}, nh={nh_arch})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"{arch['name']}: final test acc={acc_arch:.6f}")


#%%
if len(arch_test_acc) > 0:
    best_idx = int(np.argmax(arch_test_acc))
    print("\nFinal accuracy summary:")
    for name, acc in zip(arch_names, arch_test_acc):
        print(f"{name}: final test acc={acc:.6f}")
    print(f"Best architecture: {arch_names[best_idx]}, final test acc={arch_test_acc[best_idx]:.6f}")
# %%
