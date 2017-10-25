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

    if('cpu' in sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    training_settings = {
        'plot_err': True,
        'lb_samples': 20,
        'll_samples': 100,
        'n_hiddens': [50],
        'drop_rate': 0.5,
        'batch_size': 10,
        'learn_rate': 1e-3,
        'max_epochs': 2000,
        'early_stop': 10,
        'check_freq': 10,
    }
    

    """
    Log Result of BNN{13,50,1}
        1st Batch
            >>> BEST TEST
            >> Test lower bound = -279.92181396484375
            >> Test rmse = 2.7051913738250732
            >> Test log_likelihood = -2.573483943939209
        2nd Batch
            >>> BEST TEST
            >> Test lower bound = -414.76837158203125
            >> Test rmse = 3.648686170578003
            >> Test log_likelihood = -2.7757389545440674
        3rd Batch
            >>> BEST TEST
            >> Test lower bound = -462.6261291503906
            >> Test rmse = 3.4975883960723877
            >> Test log_likelihood = -2.750476360321045
        4th Batch
            >>> BEST TEST
            >> Test lower bound = -629.9293823242188
            >> Test rmse = 4.956381320953369
            >> Test log_likelihood = -3.2051596641540527
        5th Batch
            >>> BEST TEST
            >> Test lower bound = -479.79803466796875
            >> Test rmse = 3.3094685077667236
            >> Test log_likelihood = -2.7854971885681152
    """
    # run_bnn_experiment('Boston Housing', load_data(5), **training_settings)
    
    """
    Log Result of VAFNN{13,50,1}
        1st Batch
            >>> BEST TEST
            >> Test lower bound = -310.2062072753906
            >> Test rmse = 2.5632693767547607
            >> Test log_likelihood = -2.6489861011505127
        2nd Batch
            >>> BEST TEST
            >> Test lower bound = -414.76837158203125
            >> Test rmse = 3.648686170578003
            >> Test log_likelihood = -2.7757389545440674
        3rd Batch
            >>> BEST TEST
            >> Test lower bound = -462.6261291503906
            >> Test rmse = 3.4975883960723877
            >> Test log_likelihood = -2.750476360321045
        4th Batch
            >>> BEST TEST
            >> Test lower bound = -629.9293823242188
            >> Test rmse = 4.956381320953369
            >> Test log_likelihood = -3.2051596641540527
        5th Batch
            >>> BEST TEST
            >> Test lower bound = -479.79803466796875
            >> Test rmse = 3.3094685077667236
            >> Test log_likelihood = -2.7854971885681152
    """
    run_vafnn_experiment('Boston Housing', load_data(5), **training_settings)