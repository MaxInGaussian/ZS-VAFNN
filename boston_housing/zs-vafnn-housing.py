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

from expt_vafnn import run_vafnn_experiment


DATA_PATH = 'housing.data'

def load_data(n_folds):
    import pandas as pd
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep="[ ^]+")
    data = data.as_matrix().astype(np.float32)
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
        X_train, y_train = X[train_inds].copy(), y[train_inds].ravel()
        X_test, y_test = X[test_inds].copy(), y[test_inds].ravel()
        train_test_set.append([X_train, y_train, X_test, y_test])
    return train_test_set

if __name__ == '__main__':
    training_settings = {
        'lb_samples': 20,
        'll_samples': 100,
        'n_hiddens': [100],
        'drop_rate': 0.5,
        'learn_rate': 1e-3,
    }
    run_vafnn_experiment('Boston Housing', load_data(10), **training_settings)