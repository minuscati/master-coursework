#%%
import os
import sys
import numpy as np
import math

# Make stdout UTF-8 so we can print special chars from the book on Windows.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# =====================================================================
# 0.1  Read in the data
# =====================================================================
book_dir   = os.path.dirname(__file__) + '/'
book_fname = book_dir + 'goblet_book.txt'

with open(book_fname, 'r', encoding='utf-8') as fid:
    book_data = fid.read()

unique_chars = sorted(list(set(book_data)))
K = len(unique_chars)

char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

print(f'book length = {len(book_data)} chars')
print(f'K (unique chars) = {K}')

#%%
def chars_to_onehot(chars, char_to_ind=char_to_ind, K=K):
    tau = len(chars)
    X = np.zeros((K, tau), dtype=np.float64)
    idx = np.fromiter((char_to_ind[c] for c in chars), dtype=np.int64, count=tau)
    X[idx, np.arange(tau)] = 1.0
    return X


def onehot_to_chars(Y, ind_to_char=ind_to_char):
    inds = np.argmax(Y, axis=0)
    return ''.join(ind_to_char[int(i)] for i in inds)


# =====================================================================
# 0.2  Hyper-parameters & RNN initialization  (column-wise storage)
# =====================================================================
m          = 100
eta        = 1e-3
seq_length = 25

rng = np.random.default_rng(seed=42)


def InitializeRNN(K, m, rng):
    RNN = {}
    RNN['b'] = np.zeros((m, 1))
    RNN['c'] = np.zeros((K, 1))
    RNN['U'] = (1.0 / np.sqrt(2 * K)) * rng.standard_normal(size=(m, K))
    RNN['W'] = (1.0 / np.sqrt(2 * m)) * rng.standard_normal(size=(m, m))
    RNN['V'] = (1.0 / np.sqrt(m))     * rng.standard_normal(size=(K, m))
    return RNN


RNN = InitializeRNN(K, m, rng)

for k, v in RNN.items():
    print(f"RNN['{k}'] shape = {v.shape}")


X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length + 1]
X = chars_to_onehot(X_chars)
Y = chars_to_onehot(Y_chars)
print(f'X shape = {X.shape}, Y shape = {Y.shape}')
print(f'recovered X_chars: {onehot_to_chars(X) == X_chars}')


#%%
# =====================================================================
# 0.3  Synthesize text from a (randomly initialized) RNN
# =====================================================================
def softmax(O):
    O_shift = O - np.max(O, axis=0, keepdims=True)
    expO = np.exp(O_shift)
    return expO / np.sum(expO, axis=0, keepdims=True)


def SynthesizeText(RNN, h0, x0, n, rng):
    b, c = RNN['b'], RNN['c']
    U, W, V = RNN['U'], RNN['W'], RNN['V']
    K = c.shape[0]

    Y = np.zeros((K, n), dtype=np.float64)
    h = h0
    x = x0
    for t in range(n):
        a = W @ h + U @ x + b          # (m, 1)
        h = np.tanh(a)                 # (m, 1)
        o = V @ h + c                  # (K, 1)
        p = softmax(o)                 # (K, 1)

        # inverse-CDF sampling from categorical distribution p
        cp = np.cumsum(p, axis=0)
        a_rand = rng.uniform(size=1)
        ii = int(np.argmax(cp - a_rand > 0))

        Y[ii, t] = 1.0
        x = Y[:, t:t + 1]              
    return Y, h


# Demo: synthesize 200 chars from the random RNN
h0_demo = np.zeros((m, 1))
x0_demo = chars_to_onehot('.')                    # bootstrap with a full-stop
Y_demo, _ = SynthesizeText(RNN, h0_demo, x0_demo, n=200, rng=rng)
print('--- 200 chars sampled from random RNN ---')
print(onehot_to_chars(Y_demo))
print('-' * 50)

# %%
# =====================================================================
# 0.4  Forward-pass & Backward-pass
# =====================================================================
X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length + 1]
X = chars_to_onehot(X_chars)        # (K, tau)
Y = chars_to_onehot(Y_chars)        # (K, tau)
h0 = np.zeros((m, 1))


def ComputeLoss(P, Y):
    tau = P.shape[1]
    return -np.sum(Y * np.log(P + 1e-12)) / tau


def ForwardPass(X, Y, h0, RNN):
    b, c = RNN['b'], RNN['c']
    U, W, V = RNN['U'], RNN['W'], RNN['V']
    tau = X.shape[1]
    m_ = h0.shape[0]
    K_ = c.shape[0]

    A = np.zeros((m_, tau))
    H = np.zeros((m_, tau))
    O = np.zeros((K_, tau))

    h = h0
    for t in range(tau):
        x_t = X[:, t:t + 1]
        a_t = W @ h + U @ x_t + b          # (m, 1)
        h   = np.tanh(a_t)                 # (m, 1)
        o_t = V @ h + c                    # (K, 1)

        A[:, t:t + 1] = a_t
        H[:, t:t + 1] = h
        O[:, t:t + 1] = o_t

    P = softmax(O)                         # (K, tau), column-wise softmax
    loss = ComputeLoss(P, Y)

    cache = {'X': X, 'A': A, 'H': H, 'P': P, 'h0': h0}
    return loss, cache


# Sanity check on the debug sequence
loss0, cache0 = ForwardPass(X, Y, h0, RNN)
print(f'loss on first {seq_length} chars (random RNN) = {loss0:.4f}')
print(f'expected ~ log(K) = {np.log(K):.4f}  (uniform prediction baseline)')

#%%
def BackwardPass(Y, cache, RNN):
    X, H, P, h0_in = cache['X'], cache['H'], cache['P'], cache['h0']
    W, V = RNN['W'], RNN['V']
    m_, tau = H.shape

    dO = (P - Y) / tau                                        # (K, tau)

    grad_V = dO @ H.T                                          # (K, m)
    grad_c = np.sum(dO, axis=1, keepdims=True)                # (K, 1)

    dA = np.zeros((m_, tau))
    da_next = np.zeros((m_, 1))                                # dL/da_{tau+1} = 0
    for t in range(tau - 1, -1, -1):
        dh_t = V.T @ dO[:, t:t + 1] + W.T @ da_next            # (m, 1)
        h_t = H[:, t:t + 1]
        da_t = (1.0 - h_t * h_t) * dh_t                        # tanh' = 1 - h^2
        dA[:, t:t + 1] = da_t
        da_next = da_t

    H_prev = np.concatenate([h0_in, H[:, :-1]], axis=1)        # (m, tau)
    grad_W = dA @ H_prev.T                                     # (m, m)
    grad_U = dA @ X.T                                          # (m, K)
    grad_b = np.sum(dA, axis=1, keepdims=True)                 # (m, 1)

    return {'b': grad_b, 'c': grad_c, 'U': grad_U, 'W': grad_W, 'V': grad_V}


# ----- Quick sanity check on shapes -----
grads = BackwardPass(Y, cache0, RNN)
for k in ['b', 'c', 'U', 'W', 'V']:
    assert grads[k].shape == RNN[k].shape, f'shape mismatch for {k}'
print('All gradient shapes match RNN parameter shapes.')


# =====================================================================
# Gradient check against PyTorch autograd 
# =====================================================================

from torch_gradient_computations_column_wise import ComputeGradsWithTorch

m_check = 10           
seq_check = 25

rng_check = np.random.default_rng(seed=0)
RNN_check = InitializeRNN(K, m_check, rng_check)

X_check = chars_to_onehot(book_data[0:seq_check])
Y_check = chars_to_onehot(book_data[1:seq_check + 1])
y_idx = np.array([char_to_ind[c] for c in book_data[1:seq_check + 1]])
h0_check = np.zeros((m_check, 1))

_, cache_check = ForwardPass(X_check, Y_check, h0_check, RNN_check)
grads_ana = BackwardPass(Y_check, cache_check, RNN_check)
grads_torch = ComputeGradsWithTorch(X_check, y_idx, h0_check, RNN_check)

print('\n--- gradient check (analytic vs PyTorch, m=10, tau=25) ---')
for k in ['b', 'c', 'U', 'W', 'V']:
    abs_err = np.abs(grads_ana[k] - grads_torch[k]).max()
    denom = max(np.abs(grads_torch[k]).max(), 1e-12)
    rel_err = abs_err / denom
    print(f"  {k}: max abs err = {abs_err:.3e}, max rel err = {rel_err:.3e}")
# %%
# =====================================================================
import copy
import time

def EvaluateOnText(RNN, text_data, seq_length):
    n = len(text_data)
    if n <= seq_length:
        return np.nan

    hprev = np.zeros((RNN['b'].shape[0], 1))
    losses = []
    e = 0
    while e + seq_length < n:
        X_t = chars_to_onehot(text_data[e:e + seq_length], char_to_ind, K)
        Y_t = chars_to_onehot(text_data[e + 1:e + seq_length + 1], char_to_ind, K)
        loss, cache = ForwardPass(X_t, Y_t, hprev, RNN)
        losses.append(loss)
        hprev = cache['H'][:, -1:]
        e += seq_length
    return float(np.mean(losses)) if losses else np.nan

def MiniBatchAdam(
    book_data, char_to_ind, K, m, seq_length,
    GDparams, RNN, rng,
    print_every=1_000, synth_every=10_000, synth_len=200,
    val_data=None, val_every=10_000,
    batch_size=1,
    target_updates=None,
):

    n = len(book_data)
    B = int(batch_size)
    if B < 1:
        raise ValueError('batch_size must be >= 1')

    edges = np.linspace(0, n, B + 1, dtype=np.int64)
    streams = [(int(edges[i]), int(edges[i + 1])) for i in range(B)]

    # Same window count as ASS4 single-stream: (L-1)//seq_length per stream; step
    # count limited by the shortest stream so every stream stays in bounds.
    n_steps_per_stream = []
    for sb, se in streams:
        L_b = se - sb
        n_steps_per_stream.append(max(0, (L_b - 1) // seq_length))
    n_step = min(n_steps_per_stream)
    if n_step == 0:
        raise ValueError(
            f'batch_size={B} splits text into segments too short for seq_length={seq_length}; '
            'reduce batch_size or seq_length.'
        )

    n_epochs_cfg = GDparams['n_epochs']
    # Scheme A: stop after exactly `target_updates` Adam steps (fair across batch_size).
    if target_updates is not None:
        max_epochs = max(
            n_epochs_cfg,
            int(np.ceil(float(target_updates) / float(n_step))) + 1,
        )
    else:
        max_epochs = n_epochs_cfg

    eta_const = float(GDparams['eta'])
    beta1 = GDparams.get('beta1', 0.9)
    beta2 = GDparams.get('beta2', 0.999)
    eps   = GDparams.get('eps',   1e-8)
    clip  = GDparams.get('clip',  5.0)

    # ----- analogue of "trained_net = copy.deepcopy(net_params)" in ASS3 -----
    trained_RNN = copy.deepcopy(RNN)
    M = {k: np.zeros_like(v) for k, v in trained_RNN.items()}
    V = {k: np.zeros_like(v) for k, v in trained_RNN.items()}
    global_step = 0

    smooth_loss = None
    smooth_hist = []
    text_samples = []                          # (iter, text) for report iii
    val_hist = []                              # (iter, val_loss)

    print(f'MiniBatchAdam: batch_size={B}, streams ~ {[se - sb for sb, se in streams]} chars, '
          f'n_step per epoch = {n_step}')
    if target_updates is not None:
        print(f'  target_updates = {target_updates}  (max_epochs cap = {max_epochs})')

    # ----- Sample once BEFORE training (the "iter=0" entry for report iii) -----
    h0_init = np.zeros((m, 1))
    Y_init, _ = SynthesizeText(trained_RNN, h0_init, chars_to_onehot('.'),
                               synth_len, rng)
    text_samples.append((0, onehot_to_chars(Y_init)))
    print(f'\n[iter = 0  (before training)]')
    print(onehot_to_chars(Y_init))
    print('-' * 60)
    if val_data is not None:
        val0 = EvaluateOnText(trained_RNN, val_data, seq_length)
        val_hist.append((0, val0))
        print(f'[iter = 0] val_loss = {val0:.4f}')

    t_start = time.time()
    stop_training = False

    for epoch in range(max_epochs):
        pos = np.array([sb for sb, _ in streams], dtype=np.int64)
        hprev = [np.zeros((m, 1)) for _ in range(B)]

        for j in range(n_step):
            accum = {k: np.zeros_like(trained_RNN[k]) for k in trained_RNN}
            losses = []
            X_t = None

            for b in range(B):
                sb, se = streams[b]
                e_abs = int(pos[b])
                seg = book_data[sb:se]
                e_rel = e_abs - sb
                X_t = chars_to_onehot(seg[e_rel:e_rel + seq_length], char_to_ind, K)
                Y_t = chars_to_onehot(
                    seg[e_rel + 1:e_rel + seq_length + 1], char_to_ind, K)
                loss, cache = ForwardPass(X_t, Y_t, hprev[b], trained_RNN)
                grads = BackwardPass(Y_t, cache, trained_RNN)
                for k in accum:
                    accum[k] += grads[k]
                losses.append(loss)
                hprev[b] = cache['H'][:, -1:]
                pos[b] += seq_length

            inv_B = 1.0 / B
            for k in accum:
                accum[k] *= inv_B
            loss_mean = float(np.mean(losses))

            global_step += 1
            for k in trained_RNN.keys():
                g = accum[k]
                # g = np.clip(g, -clip, clip)
                M[k] = beta1 * M[k] + (1 - beta1) * g
                V[k] = beta2 * V[k] + (1 - beta2) * (g * g)
                m_hat = M[k] / (1 - beta1 ** global_step)
                v_hat = V[k] / (1 - beta2 ** global_step)
                trained_RNN[k] -= eta_const * m_hat / (np.sqrt(v_hat) + eps)

            smooth_loss = loss_mean if smooth_loss is None \
                else 0.999 * smooth_loss + 0.001 * loss_mean
            smooth_hist.append(smooth_loss)

            if global_step % print_every == 0:
                print(f'epoch {epoch + 1}/{max_epochs}  iter = {global_step:6d}'
                      f'   smooth_loss = {smooth_loss:.4f}'
                      f'   ({time.time() - t_start:.1f}s)')

            if global_step % synth_every == 0:
                # Bootstrap synth from last stream's current prefix char (matches B=1 habit).
                x0 = X_t[:, 0:1] if X_t is not None else chars_to_onehot('.', char_to_ind, K)
                Y_synth, _ = SynthesizeText(trained_RNN, np.zeros((m, 1)),
                                            x0, synth_len, rng)
                txt = onehot_to_chars(Y_synth)
                text_samples.append((global_step, txt))
                print(f'\n[iter = {global_step}]')
                print(txt)
                print('-' * 60)
            if (val_data is not None) and (global_step % val_every == 0):
                val_loss = EvaluateOnText(trained_RNN, val_data, seq_length)
                val_hist.append((global_step, val_loss))
                print(f"[iter={global_step}] val_loss={val_loss:.4f}")

            if target_updates is not None and global_step >= target_updates:
                stop_training = True
                break

        if stop_training:
            break

    print(f'\nTraining done in {time.time() - t_start:.1f}s. '
          f'total updates = {global_step}')

    metrics = {
        'smooth_hist':  smooth_hist,
        'text_samples': text_samples,
        'global_step':  global_step,
        'val_hist':     val_hist,
        'batch_size':   B,
        'target_updates': target_updates,
    }
    return trained_RNN, M, V, metrics

# =====================================================================
# Main: run the training (mirrors the "MAIN" section style of ASS3)
# =====================================================================
GDparams = {
    'n_epochs': 4,            
    'eta':      eta,
    'beta1':    0.9,
    'beta2':    0.999,
    'eps':      1e-8,
    'clip':     5.0,
}

target_updates = 100_000

results_dir = os.path.normpath(os.path.join(book_dir, '..', 'Results', 'bonus2'))
os.makedirs(results_dir, exist_ok=True)
results_dir = results_dir + os.sep
print(f'Bonus2 output directory: {results_dir}')

# Hold out the last 10% of the book as validation split
val_ratio = 0.1
split_idx = int((1.0 - val_ratio) * len(book_data))
train_data = book_data[:split_idx]
val_data = book_data[split_idx:]
print(f"train chars = {len(train_data)}, val chars = {len(val_data)}")

# Bonus2: sweep batch sizes 
import matplotlib.pyplot as plt

batch_sizes = [1, 8, 16, 32]
metrics_by_bs = {}

val_every = max(1_000, target_updates // 20)

for bs in batch_sizes:
    print('\n' + '=' * 60)
    print(f'>>> Training with batch_size = {bs}')
    print('=' * 60)

    rng_run = np.random.default_rng(seed=42)
    RNN_run = InitializeRNN(K, m, rng_run)
    trained_run, M_run, V_run, metrics_run = MiniBatchAdam(
        train_data, char_to_ind, K, m, seq_length,
        GDparams, RNN_run, rng_run,
        print_every=1_000,
        synth_every=10_000,
        synth_len=200,
        val_data=val_data,
        val_every=val_every,
        batch_size=bs,
        target_updates=target_updates,
    )
    metrics_by_bs[bs] = metrics_run

    tag = f'_bs{bs}'
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_run['smooth_hist'],
             label=f'batch_size={metrics_run["batch_size"]}')
    plt.xlabel('update step')
    plt.ylabel('smooth loss')
    plt.title(f'smooth_loss (batch_size={bs})')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir + f'smooth_loss_bonus2{tag}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'saved {results_dir}smooth_loss_bonus2{tag}.png')

    # --- Text samples log ---
    with open(results_dir + f'text_samples_bonus2{tag}.txt', 'w', encoding='utf-8') as f:
        for it, txt in metrics_run['text_samples']:
            if it == 0:
                f.write('=== iter = 0  (before training) ===\n')
            else:
                f.write(f'=== iter = {it}   '
                        f'smooth_loss = {metrics_run["smooth_hist"][it - 1]:.4f} ===\n')
            f.write(txt + '\n\n')
    print(f'saved {results_dir}text_samples_bonus2{tag}.txt')

    # --- Validation loss log ---
    with open(results_dir + f'val_loss_bonus2{tag}.txt', 'w', encoding='utf-8') as f:
        f.write('iter,val_loss\n')
        for it, vl in metrics_run['val_hist']:
            f.write(f'{it},{vl:.6f}\n')
    print(f'saved {results_dir}val_loss_bonus2{tag}.txt')


#%%
# --- Overlay: all batch sizes on one figure (smooth + val) ---
plt.figure(figsize=(9, 5))
for bs in batch_sizes:
    h = metrics_by_bs[bs]['smooth_hist']
    plt.plot(h, label=f'batch_size={bs}')
plt.xlabel('update step')
plt.ylabel('smooth loss')
plt.title('smooth_loss comparison (batch sweep)')
plt.legend()
plt.grid(True)
plt.savefig(results_dir + 'smooth_loss_bonus2_compare.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'saved {results_dir}smooth_loss_bonus2_compare.png')

plt.figure(figsize=(9, 5))
for bs in batch_sizes:
    vh = metrics_by_bs[bs]['val_hist']
    if not vh:
        continue
    steps = [it for it, _ in vh]
    vals = [vl for _, vl in vh]
    plt.plot(steps, vals, marker='o', label=f'batch_size={bs}')
plt.xlabel('update step')
plt.ylabel('validation loss')
plt.title('val_loss comparison (batch sweep)')
plt.legend()
plt.grid(True)
plt.savefig(results_dir + 'val_loss_bonus2_compare.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'saved {results_dir}val_loss_bonus2_compare.png')

