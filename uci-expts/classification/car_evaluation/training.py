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


DATA_PATH = 'car.data'

def load_data(n_folds):
    np.random.seed(314159)
    import pandas as pd
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep=",")
    data = data.sample(frac=1).dropna(axis=0)
    data = pd.get_dummies(data).as_matrix()
    X, y = data[:, :-2].astype(np.float32), data[:, -2:].astype(np.int32)
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
        'DNN', 'VIBayesNN', 'VIFourAct', 'MCDropout', 'MCFourAct'
    ]
    
    dataset = load_data(5)
    D, P = dataset[0][0].shape[1], dataset[0][1].shape[1]
    
    # Fair Model Comparison - Same Architecture & Optimization Rule
    training_settings = {
        'task': "classification",
        'save': False,
        'plot': True,
        'n_basis': 50,
        'drop_rate': 0.5,
        'train_samples': 10,
        'test_samples': 50,
        'max_iters': 100,
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
        model_names, 'Car Evaluation', dataset, **training_settings)
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
        >>> BayesNN
        >> err_rate = 0.0378 p/m 0.0058
        >> log_likelihood = -0.3359 p/m 0.0650
        >>> DropoutNN
        >> err_rate = 0.1611 p/m 0.0590
        >> log_likelihood = -0.1530 p/m 0.0137
        >>> VAFNN
        >> err_rate = 0.0732 p/m 0.0707
        >> log_likelihood = -0.1337 p/m 0.0344
    '''