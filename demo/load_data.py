''' load_data.py '''
import numpy as np
import pandas as pd

DATA_PATH = 'housing.data'

## Load and return K sets of data (each contains train, validation, and test)
def load_data(n_folds):
    np.random.seed(314159)
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep="[ ^]+")
    data = data.sample(frac=1).dropna(axis=0).as_matrix().astype(np.float32)
    X, Y = data[:, :-1], data[:, -1]
    Y = Y[:, None]
    n_data = Y.shape[0]
    n_partition = n_data//n_folds
    n_train = n_partition*(n_folds-1)
    dataset, folds = [], []
    for i in range(n_folds):
        if(i == n_folds-1):
            fold_inds = np.arange(n_data)[i*n_partition:]
        else:
            fold_inds = np.arange(n_data)[i*n_partition:(i+1)*n_partition]
        folds.append([X[fold_inds], Y[fold_inds]])
    for i in range(n_folds):
        valid_fold, test_fold = i, (i+1)%n_folds
        train_folds = np.setdiff1d(np.arange(n_folds), [test_fold, valid_fold])
        X_train = np.vstack([folds[fold][0] for fold in train_folds])
        Y_train = np.vstack([folds[fold][1] for fold in train_folds])
        X_valid, Y_valid = folds[valid_fold]
        X_test, Y_test = folds[test_fold]
        dataset.append([X_train, Y_train, X_valid, Y_valid, X_test, Y_test])
    return dataset

## Standardize the scale of input-output pairs to N(0, 1)
def scaler(data_train, data_valid, data_test):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    train_standardized = (data_train - mean)/std
    valid_standardized = (data_test - mean)/std
    test_standardized = (data_test - mean)/std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return train_standardized, valid_standardized, test_standardized, mean, std