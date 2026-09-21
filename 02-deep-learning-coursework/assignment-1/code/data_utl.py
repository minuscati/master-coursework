import pickle
import numpy as np
import pandas as pd

def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    X = data_dict[b'data'].astype(np.float64)/255.0
    X = X.transpose()
    y = data_dict[b'labels']
    Y = pd.get_dummies(y).values.astype(np.float64)
    Y = Y.transpose()
    return X, Y, y


def normalize_data(X_tr, X_val, X_te):
    mean_X = np.mean(X_tr, axis=1, keepdims=True)
    std_X = np.std(X_tr, axis=1, keepdims=True)

    X_tr = (X_tr - mean_X) / std_X
    X_val = (X_val - mean_X) / std_X
    X_te = (X_te - mean_X) / std_X

    return X_tr, X_val, X_te