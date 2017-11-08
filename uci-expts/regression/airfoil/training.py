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


DATA_PATH = 'airfoil_self_noise.dat'

def load_data(n_folds):
    np.random.seed(314159)
    import pandas as pd
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep="[ \t^]+")
    data = data.sample(frac=1).dropna(axis=0).as_matrix().astype(np.float32)
    X, y = data[:, :-1], data[:, -1]
    y = y[:, None]
    n_data = y.shape[0]
    n_partition = n_data//n_folds
    n_train = n_partition*(n_folds-1)
    train_test_set = []
    for fold in range(n_folds):
        if(fold == n_folds-1):
            test_inds = np.arange(n_data)[fold*n_partition:]
        else:
            test_inds = np.arange(n_data)[fold*n_partition:(fold+1)*n_partition]
        train_inds = np.setdiff1d(range(n_data), test_inds)
        X_train, y_train = X[train_inds], y[train_inds]
        X_test, y_test = X[test_inds], y[test_inds]
        train_test_set.append([X_train, y_train, X_test, y_test])
    return train_test_set

if __name__ == '__main__':

    if('cpu' in sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model_names = [
        'DNN', 'VIBayesNN', 'MCDropout', 'MCFourAct'
    ]
    
    train_test_set = load_data(5)
    D, P = train_test_set[0][0].shape[1], train_test_set[0][1].shape[1]
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    training_settings = {
        'task': "regression",
        'save': False,
        'plot': True,
        'n_basis': 50,
        'drop_rate': 0.5,
        'train_samples': 10,
        'test_samples': 100,
        'max_iters': 1000,
        'n_hiddens': [50, 25],
        'batch_size': 10,
        'learn_rate': 1e-3,
        'max_epochs': 1000,
        'early_stop': 5,
        'check_freq': 5,
    }
     
    for argv in sys.argv:
        if('--' == argv[:2] and '=' in argv):
            eq_ind = argv.index('=')
            setting_feature = argv[2:eq_ind]
            if(setting_feature in ['save', 'plot']):
                training_settings[setting_feature] = (argv[eq_ind+1:]=='True')
    
    print(training_settings)
    
    eval_rmses, eval_lls = run_experiment(
        model_names, 'Airfoil', train_test_set, **training_settings)
    print(eval_rmses, eval_lls)
    
    for model_name in model_names:
        rmse_mu = np.mean(eval_rmses[model_name])
        rmse_std = np.std(eval_rmses[model_name])
        ll_mu = np.mean(eval_lls[model_name])
        ll_std = np.std(eval_lls[model_name])
        print('>>> '+model_name)
        print('>> RMSE = {:.4f} \pm {:.4f}'.format(rmse_mu, rmse_std))
        print('>> NLPD = {:.4f} \pm {:.4f}'.format(ll_mu, ll_std))
    
    '''
    Result:
        >>> BayesNN
        >> RMSE = 3.4740 \pm 0.4435
        >> NLPD = 2.6974 \pm 0.1334
        >>> DropoutNN
        >> RMSE = 3.3841 \pm 0.3789
        >> NLPD = 2.8674 \pm 0.1207
        >>> VAFNN
        >> RMSE = 3.2383 \pm 0.2281
        >> NLPD = 2.6131 \pm 0.0929
    '''