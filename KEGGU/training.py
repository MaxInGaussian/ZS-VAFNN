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

from expt_bnn import run_bnn_experiment
from expt_vafnn import run_vafnn_experiment


DATA_PATH = 'Reaction Network (Undirected).data'

def load_data(n_folds):
    import pandas as pd
    data = pd.DataFrame.from_csv(path=DATA_PATH, header=None, index_col=0)
    data = data.as_matrix().astype(np.float32)
    X, y = np.hstack((data[:, :-3], data[:, -2:])), data[:, -3]
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
        X_train, y_train = X[train_inds].copy(), y[train_inds].ravel()
        X_test, y_test = X[test_inds].copy(), y[test_inds].ravel()
        train_test_set.append([X_train, y_train, X_test, y_test])
    return train_test_set

if __name__ == '__main__':

    if('cpu' in sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    training_settings = {
        'plot_err': True,
        'lb_samples': 20,
        'll_samples': 100,
        'n_hiddens': [50],
        'drop_rate': 0.5,
        'batch_size': 50,
        'learn_rate': 1e-3,
        'max_epochs': 2000,
        'early_stop': 10,
        'check_freq': 10,
    }
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    run_bnn_experiment('KEGGU', load_data(10), **training_settings)
    run_vafnn_experiment('KEGGU', load_data(10), **training_settings)