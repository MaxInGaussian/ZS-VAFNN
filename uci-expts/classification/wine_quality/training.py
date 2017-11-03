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
    
    model_names = ['BayesNN', 'DropoutNN', 'VAFNN']
    
    train_test_set = load_data(5)
    D, P = train_test_set[0][0].shape[1], train_test_set[0][1].shape[1]
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    training_settings = {
        'task': "classification",
        'save': False,
        'plot': True,
        'n_basis': 50,
        'drop_rate': 0.5,
        'lb_samples': 10,
        'll_samples': 50,
        'n_hiddens': [50],
        'batch_size': 10,
        'learn_rate': 1e-3,
        'max_epochs': 500,
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
        model_names, 'Wine Quality', train_test_set, **training_settings)
    print(eval_err_rates, eval_lls)
    #{'BayesNN': [0.44092211, 0.45516115, 0.46715629, 0.43888211, 0.43265733], 'DropoutNN': [0.44981638, 0.44394124, 0.4481436, 0.42766216, 0.4279021], 'VAFNN': [0.44940841, 0.42811096, 0.4644635, 0.4272134, 0.41489512]} {'BayesNN': [-1.0454601, -1.0824183, -1.101755, -1.0656945, -1.0338538], 'DropoutNN': [-1.0423433, -1.0654633, -1.0732477, -1.0289091, -1.0281358], 'VAFNN': [-1.0262312, -1.0475413, -1.1155545, -1.0521262, -1.0072927]}
    
    for model_name in model_names:
        errt_mu = np.mean(eval_err_rates[model_name])
        errt_std = np.std(eval_err_rates[model_name])
        ll_mu = np.mean(eval_lls[model_name])
        ll_std = np.std(eval_lls[model_name])
        print('>>> '+model_name)
        print('>> err_rate = {:.4f} p/m {:.4f}'.format(errt_mu, errt_std))
        print('>> log_likelihood = {:.4f} p/m {:.4f}'.format(ll_mu, ll_std))
    
    '''
    Result:
        >>> BayesNN
        >> rmse = 3.8680 p/m 0.4681
        >> log_likelihood = -2.7765 p/m 0.0828
        >>> DropoutNN
        >> rmse = 4.3249 p/m 0.2355
        >> log_likelihood = -2.7864 p/m 0.2238
        >>> VAFNN
        >> rmse = 3.4084 p/m 0.3560
        >> log_likelihood = -2.6322 p/m 0.1433
    '''