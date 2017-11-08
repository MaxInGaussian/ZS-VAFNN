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
sys.path.append("../../")

import os
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from six.moves import range, zip
import numpy as np
import zhusuan as zs

import six
import gzip
from six.moves import cPickle as pickle

from expt import run_experiment


DATA_PATH = 'mnist.pkl.gz'

def load_data(n_folds):
    np.random.seed(1234)
    def to_one_hot(y, n_class):
        y_onehot = np.zeros((y.shape[0], n_class))
        y_onehot[np.arange(y.shape[0]), y] = 1
        return y_onehot
    f = gzip.open(DATA_PATH, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    X_train, y_train = train_set[0], train_set[1]
    X_valid, y_valid = valid_set[0], valid_set[1]
    X_test, y_test = test_set[0], test_set[1]
    X_train = np.vstack([X_train, X_valid]).astype('float32')
    y_train = np.concatenate([y_train, y_valid])
    return [[X_train, to_one_hot(y_train, 10), X_test, to_one_hot(y_test, 10)]]


if __name__ == '__main__':

    if('cpu' in sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model_names = [
        'VIBayesNN', 'MCDropout', 'MCFourAct'
    ]
    
    train_test_set = load_data(5)
    D, P = train_test_set[0][0].shape[1], train_test_set[0][1].shape[1]
    
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
        'n_hiddens': [100],
        'batch_size': 50,
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
        model_names, 'MNIST', train_test_set, **training_settings)
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
        >>> BayesNN
        >> err_rate = 0.1458 p/m 0.0036
        >> log_likelihood = -0.3113 p/m 0.0041
        >>> DropoutNN
        >> err_rate = 0.1539 p/m 0.0035
        >> log_likelihood = -0.3413 p/m 0.0036
        >>> VAFNN
        >> err_rate = 0.1454 p/m 0.0039
        >> log_likelihood = -0.3203 p/m 0.0052
    '''