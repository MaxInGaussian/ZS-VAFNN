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


DATA_PATH = 'spambase.data'

def load_data(n_folds):
    np.random.seed(314159)
    import pandas as pd
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep=",")
    data = data.sample(frac=1).dropna(axis=0)
    data = pd.get_dummies(data).as_matrix()
    X, y = data[:, :-1].astype(np.float32), data[:, -1].astype(np.int32)
    y = np.hstack((y[:, None], 1-y[:, None]))
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
        'VIBayesNN', 'MCDropout', 'MCFourAct'
    ]
    
    train_test_set = load_data(5)
    N, D = train_test_set[0][0].shape
    T, P = train_test_set[0][-1].shape
    print("N = %d, D = %d, T = %d, P = %d"%(N, D, T, P))
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    training_settings = {
        'task': "classification",
        'save': False,
        'plot': True,
        'n_basis': 50,
        'drop_rate': 0.5,
        'train_samples': 10,
        'test_samples': 50,
        'max_iters': 1000,
        'n_hiddens': [50, 25],
        'batch_size': 10,
        'learn_rate': 1e-3,
        'max_epochs': 1000,
        'early_stop': 5,
        'check_freq': 10,
    }
     
    for argv in sys.argv:
        if('--' == argv[:2] and '=' in argv):
            eq_ind = argv.index('=')
            setting_feature = argv[2:eq_ind]
            if(setting_feature in ['save', 'plot']):
                training_settings[setting_feature] = (argv[eq_ind+1:]=='True')

    print(training_settings)

    eval_err_rates, eval_lls = run_experiment(
        model_names, 'Spambase', train_test_set, **training_settings)
    print(eval_err_rates, eval_lls)
    
    for model_name in model_names:
        errt_mu = np.mean(eval_err_rates[model_name])
        errt_std = np.std(eval_err_rates[model_name])
        ll_mu = np.mean(eval_lls[model_name])
        ll_std = np.std(eval_lls[model_name])
        print('>>> '+model_name)
        print('>> CERR = {:.4f} \pm {:.4f}'.format(errt_mu, errt_std))
        print('>> AUC = {:.4f} \pm {:.4f}'.format(ll_mu, ll_std))
    
    '''
    Result:
        >>> VIBayesNN
        >> CERR = 0.0711 \pm 0.0048
        >> AUC = 0.9680 \pm 0.0006
        >>> MCDropout
        >> CERR = 0.0815 \pm 0.0043
        >> AUC = 0.9591 \pm 0.0019
        >>> MCFourAct
        >> CERR = 0.0665 \pm 0.0044
        >> AUC = 0.9665 \pm 0.0014
    '''
