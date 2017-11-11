# Copyright 2017 Max W. Y. Lam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
sys.path.append("../../../")

import os
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from six.moves import range, zip
import numpy as np
import zhusuan as zs

from expt import run_experiment


DATA_PATH = 'winequality.csv'

def load_data(n_folds):
    np.random.seed(314159)
    def to_one_hot(y):
        labels = np.unique(y).tolist()
        y_onehot = np.zeros((y.shape[0], len(labels)))
        for i in range(y.shape[0]):
            y_onehot[i, labels.index(y[i])] = 1
        return y_onehot, labels
    import pandas as pd
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=0, index_col=None, sep=";")
    data = data.sample(frac=1).dropna(axis=0).as_matrix()
    X = data[:, :-1].astype(np.float32)
    y, labels = to_one_hot(data[:, -1].astype(np.int32))
    n_data = y.shape[0]
    n_partition = n_data//n_folds
    n_train = n_partition*(n_folds-1)
    dataset, folds = [], []
    for i in range(n_folds):
        if(i == n_folds-1):
            fold_inds = np.arange(n_data)[i*n_partition:]
        else:
            fold_inds = np.arange(n_data)[i*n_partition:(i+1)*n_partition]
        folds.append([X[fold_inds], y[fold_inds]])
    for i in range(n_folds):
        valid_fold, test_fold = i, (i+1)%n_folds
        train_folds = np.setdiff1d(np.arange(n_folds), [test_fold, valid_fold])
        X_train = np.vstack([folds[fold][0] for fold in train_folds])
        y_train = np.vstack([folds[fold][1] for fold in train_folds])
        X_valid, y_valid = folds[valid_fold]
        X_test, y_test = folds[test_fold]
        dataset.append([X_train, y_train, X_valid, y_valid, X_test, y_test])
    return dataset


if __name__ == '__main__':

    if('cpu' in sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model_names = [
        'DNN', 'VIBayesNN', 'MCDropout', 'MCSSA', 'MCSSADropout'
    ]
    
    dataset = load_data(5)
    N, D = dataset[0][0].shape
    T, P = dataset[0][-1].shape
    print("N = %d, D = %d, T = %d, P = %d"%(N, D, T, P))
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    training_settings = {
        'task': "classification",
        'save': False,
        'plot': True,
        'n_basis': 50,
        'drop_rate': 0.15,
        'train_samples': 10,
        'test_samples': 100,
        'max_iters': 1000,
        'n_hiddens': [50, 25],
        'batch_size': 10,
        'learn_rate': 1e-3,
        'max_epochs': 1500,
        'early_stop': 10,
        'check_freq': 5,
    }
     
    for argv in sys.argv:
        if('--' == argv[:2] and '=' in argv):
            eq_ind = argv.index('=')
            setting_feature = argv[2:eq_ind]
            setting_value = argv[eq_ind+1:]
            if(setting_feature in ['save', 'plot']):
                training_settings[setting_feature] = (setting_value=='True')
            if(setting_feature == 'model'):
                model_names = [setting_value]
    
    print(training_settings)

    eval_err_rates, eval_lls = run_experiment(
        model_names, 'Wine Quality', dataset, **training_settings)
    print(eval_err_rates, eval_lls)
    
    for model_name in model_names:
        errt_mu = np.mean(eval_err_rates[model_name])
        errt_std = np.std(eval_err_rates[model_name])
        ll_mu = np.mean(eval_lls[model_name])
        ll_std = np.std(eval_lls[model_name])
        print('>>> '+model_name)
        print('>> ERRT = {:.4f} \pm {:.4f}'.format(errt_mu, 1.96*errt_std))
        print('>> AUC = {:.4f} \pm {:.4f}'.format(ll_mu, 1.96*ll_std))
    
    '''
    Result:
        >>> VIBayesNN
        >> ERRT = 0.4465 \pm 0.0116
        >> AUC = 0.9024 \pm 0.0005
        >>> MCDropout
        >> ERRT = 0.4702 \pm 0.0177
        >> AUC = 0.8744 \pm 0.0018
        >>> MCFourAct
        >> ERRT = 0.4079 \pm 0.0070
        >> AUC = 0.9224 \pm 0.0012
    '''
