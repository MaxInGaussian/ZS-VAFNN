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
sys.path.append("../")

import os
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from six.moves import range, zip
import numpy as np
import zhusuan as zs

from expt import run_experiment


DATA_PATH = 'parkinsons_updrs.data'

def load_data(n_folds):
    import pandas as pd
    data = pd.DataFrame.from_csv(path=DATA_PATH, header=0, index_col=0)
    data = data.sample(frac=1).dropna(axis=0).as_matrix().astype(np.float32)
    X, y = np.hstack((data[:, :4], data[:, 5:])), data[:, 4]
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
    
    model_names = ['VAFNN', 'DropoutNN', 'BayesNN']
    
    train_test_set = load_data(5)
    D, P = train_test_set[0][0].shape[1], train_test_set[0][1].shape[1]
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    training_settings = {
        'save': False,
        'plot_err': True,
        'drop_rate': 0.5,
        'lb_samples': 20,
        'll_samples': 100,
        'n_basis': 50,
        'n_hiddens': [100],
        'batch_size': 10,
        'learn_rate': 1e-2,
        'max_epochs': 10000,
        'early_stop': 5,
        'check_freq': 5,
    }

    eval_mses, eval_lls = run_experiment(
        model_names, 'Parkinsons', load_data(5), **training_settings)
    print(eval_mses, eval_lls)