#%% Exercise 1
import numpy as np
import os
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
debug_file = os.path.join(_script_dir, "debug_info.npz")
load_data = np.load(debug_file)


X = load_data['X'] #3072*5
Fs = load_data['Fs'] #4*4*3*2

# %%
n = X.shape[1] #5
f = Fs.shape[0] #4
nf = Fs.shape[3] #2
X_ims = np.transpose(X.reshape((32,32,3,n), order='F'), (1,0,2,3))

#%%
out_h = 32//f
out_w = 32//f
conv_out = np.zeros((out_h, out_w, nf, n))

#%% loop
for i in range(n):
    for k in range(nf):
        Fk = Fs[:,:,:,k]
        for  r in range(out_h):
            r0 = r*f
            r1 = r0 + f
            for c in range(out_w):
                c0 = c*f
                c1 = c0 + f
                X_patch = X_ims[r0:r1,c0:c1,:,i]
                conv_out[r,c,k,i] = np.sum(np.multiply(X_patch, Fk))
#%% Compare
conv_gt = load_data['conv_outputs']
print("conv_outputs shape:", conv_out.shape)
print("conv_gt shape:", conv_gt.shape)
max_abs_err = np.max(np.abs(conv_out - conv_gt))
print("Max absolute error:", max_abs_err)

#%% Vectorization
# build MX
n_p = out_h * out_w
MX = np.zeros((n_p, f*f*3, n), dtype=np.float64)
for i in range(n):
    l = 0
    for r in range(out_h):
        r0 = r*f
        r1 = r0 + f
        for c in range(out_w):
            c0 = c*f
            c1 = c0 + f
            X_patch = X_ims[r0:r1,c0:c1,:,i]
            MX[l, :, i] = X_patch.reshape((1, f * f * 3), order="C")
            l += 1

# matmul
Fs_flat = Fs.reshape((f*f*3, nf), order = 'C')
conv_outputs_mat = np.zeros((n_p, nf, n), dtype=np.float64)
for i in range(n):
    conv_outputs_mat[:,:,i] = np.matmul(MX[:,:,i], Fs_flat)

# vec compare with loop 
conv_outputs_flat = conv_out.reshape((n_p,nf,n), order = 'C')
print("err(mat vs slow):", np.max(np.abs(conv_outputs_mat - conv_outputs_flat)))

# compare with vec, loop 
conv_ein = np.einsum('ijn,jl->iln', MX, Fs_flat, optimize=True)  
print("err(ein vs mat):", np.max(np.abs(conv_ein - conv_outputs_mat)))
print("err(ein vs slow):", np.max(np.abs(conv_ein - conv_outputs_flat)))

#--------------------------------------------------
#%% Exercise 2
def ReLu(x):
    return np.maximum(0, x)

def softmax(S):
    S_shift = S - np.max(S, axis=0, keepdims=True)
    nominators = np.exp(S_shift)
    denominators = np.sum(nominators, axis=0, keepdims=True)
    return nominators / denominators

def forward_pass(MX, Fs_flat, W1, b1, W2, b2, b_conv):
    n_p, _, n = MX.shape
    nf = Fs_flat.shape[1]

    H1 = np.einsum('ijn,jl->iln', MX, Fs_flat, optimize=True) + b_conv
    conv_flat = ReLu(H1.reshape((n_p*nf, n), order='C')) 
    X1_pre = np.matmul(W1, conv_flat) + b1
    X1 = ReLu(X1_pre)
    S = np.matmul(W2, X1) + b2
    P = softmax(S)

    Mid_params = {
        'H1': H1, #(n_p, nf, n)
        'conv_flat': conv_flat, #(n_p*nf, n)
        'X1_pre': X1_pre, #(nh, n)
        'X1': X1, #(nh, n)
        'S': S, #(10, n)
    }
    return P, Mid_params

def backward_pass(P, MX, Y, Fs_Flat, W1, b1, W2, b2, b_conv, lam, Mid_params):
    H1, conv_flat, X1_pre, X1, S = Mid_params.values()
    n = Y.shape[1]
    n_p = MX.shape[0]
    nf = Fs_Flat.shape[1]

    G = P - Y #(10, n)
    grad_W2 = (1/n) * np.matmul(G, X1.T) + 2 * lam * W2
    grad_b2 = (1/n) * np.sum(G, axis=1, keepdims=True)

    G = np.matmul(W2.T, G) #(nh, n)
    G = G * (X1_pre > 0) #(nh, n)

    grad_W1 = (1/n) * np.matmul(G, conv_flat.T) + 2 * lam * W1
    grad_b1 = (1/n) * np.sum(G, axis=1, keepdims=True)
    
    G_batch = np.matmul(W1.T, G)
    G_batch = G_batch * (conv_flat > 0)

    GG = G_batch.reshape((n_p, nf, n), order='C')
    grad_b_conv = ((1 / n) * np.sum(GG, axis=(0, 2), keepdims=False)).reshape((-1, 1))

    Mxt = np.transpose(MX, (1,0,2))
    grad_Fs_flat = (1/n) * np.einsum('ijn,jln->il', Mxt, GG, optimize=True)
    grad_Fs_flat += 2 * lam * Fs_Flat

    grad_params = {
        'grad_W2': grad_W2,
        'grad_b2': grad_b2,
        'grad_W1': grad_W1,
        'grad_b1': grad_b1,
        'grad_Fs_flat': grad_Fs_flat,
        'grad_b_conv': grad_b_conv,
    }
    return grad_params


def max_err(a, b, name):
    err = np.max(np.abs(a - b))
    print(f"{name:20s} shape={a.shape}, err={err:.3e}")


def _get_b_conv(load_data, nf, dtype=np.float64):
    if "b_conv" in load_data.files:
        return np.asarray(load_data["b_conv"], dtype=dtype)
    return np.zeros((nf, 1), dtype=dtype)


def run_debug_check(load_data, MX, Fs):
    W1 = load_data['W1']
    b1 = load_data['b1']
    W2 = load_data['W2']
    b2 = load_data['b2']
    Y = load_data['Y']

    f, _, _, nf = Fs.shape
    Fs_flat = Fs.reshape((f*f*3, nf), order = 'C')
    b_conv = _get_b_conv(load_data, nf, dtype=np.asarray(W1).dtype)

    # forward pass
    P, Mid_params = forward_pass(MX, Fs_flat, W1, b1, W2, b2, b_conv)
    max_err(Mid_params['conv_flat'], load_data['conv_flat'], 'conv_flat')
    max_err(Mid_params['X1'], load_data['X1'], 'X1')
    max_err(P, load_data['P'], 'P')

    # backward pass
    lam = 0
    grad_params = backward_pass(P, MX, Y, Fs_flat, W1, b1, W2, b2, b_conv, lam, Mid_params)
    max_err(grad_params['grad_Fs_flat'], load_data['grad_Fs_flat'], 'grad_Fs_flat')
    if "grad_b_conv" in load_data.files:
        max_err(grad_params["grad_b_conv"], load_data["grad_b_conv"], "grad_b_conv")

def compare_grad(name, g_np, g_th):
    g_th_np = g_th.detach().cpu().numpy()
    abs_err = np.max(np.abs(g_np - g_th_np))
    rel_err = abs_err / max(1e-12, np.max(np.abs(g_np)) + np.max(np.abs(g_th_np)))
    print(f"{name:12s} shape={g_np.shape}, abs_err={abs_err:.3e}, rel_err={rel_err:.3e}")

def torch_check(load_data, MX, Fs, lam=0.0):
    W1 = load_data['W1'].astype(np.float64)
    b1 = load_data['b1'].astype(np.float64)
    W2 = load_data['W2'].astype(np.float64)
    b2 = load_data['b2'].astype(np.float64)
    Y = load_data['Y'].astype(np.float64)

    f, _, _, nf = Fs.shape
    Fs_flat = Fs.reshape((f * f * 3, nf), order='C').astype(np.float64)
    b_conv = _get_b_conv(load_data, nf, dtype=np.float64)
    MX_np = MX.astype(np.float64)

    # numpy grads
    P_np, Mid_params = forward_pass(MX_np, Fs_flat, W1, b1, W2, b2, b_conv)
    grads_np = backward_pass(P_np, MX_np, Y, Fs_flat, W1, b1, W2, b2, b_conv, lam, Mid_params)

    # torch tensors
    MX_t = torch.tensor(MX_np, dtype=torch.float64)
    Y_t = torch.tensor(Y, dtype=torch.float64)
    Fs_t = torch.tensor(Fs_flat, dtype=torch.float64, requires_grad=True)
    W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
    b_conv_t = torch.tensor(b_conv, dtype=torch.float64, requires_grad=True)

    # torch forward
    H1_t = torch.einsum('ijn,jl->iln', MX_t, Fs_t) + b_conv_t  # (n_p, nf, n)
    n_p, nf_t, n = H1_t.shape
    conv_flat_t = torch.relu(H1_t.reshape(n_p * nf_t, n))
    X1_pre_t = W1_t @ conv_flat_t + b1_t
    X1_t = torch.relu(X1_pre_t)
    S_t = W2_t @ X1_t + b2_t
    P_t = torch.softmax(S_t, dim=0)

    # mean CE + L2
    loss_ce = -torch.sum(Y_t * torch.log(P_t + 1e-12)) / n
    loss_reg = lam * (torch.sum(W1_t * W1_t) + torch.sum(W2_t * W2_t) + torch.sum(Fs_t * Fs_t))
    loss = loss_ce + loss_reg
    loss.backward()

    print("\n[Torch gradient check]")
    compare_grad('W2', grads_np['grad_W2'], W2_t.grad)
    compare_grad('b2', grads_np['grad_b2'], b2_t.grad)
    compare_grad('W1', grads_np['grad_W1'], W1_t.grad)
    compare_grad('b1', grads_np['grad_b1'], b1_t.grad)
    compare_grad('Fs_flat', grads_np['grad_Fs_flat'], Fs_t.grad)
    compare_grad('b_conv', grads_np['grad_b_conv'], b_conv_t.grad)
#%%
run_debug_check(load_data, MX, Fs)
torch_check(load_data, MX, Fs, lam=0.0)


#%%
# print(X.shape)
# print(Fs.shape)
print(X_ims.shape)
# %%
