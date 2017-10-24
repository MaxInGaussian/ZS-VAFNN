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


############################ Data Setting ############################
DATA_PATH = 'Relation Network (Directed).data'
BEST_MODEL_PATH = 'best_model.pkl'
n_folds = 10

def load_data(n_folds=n_folds):
    import pandas as pd
    data = pd.DataFrame.from_csv(path=DATA_PATH, header=None, index_col=0)
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

############################ Model Setting ############################
@zs.reuse('model')
def variational_activation_functions_neural_networks(
    observed, X, D, layer_sizes, drop_rate, n_samples, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        f = tf.expand_dims(tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1]), 3)
        kern_logscale = tf.get_variable('kern_logscale',
            shape=[len(layer_sizes)-2], initializer=tf.constant_initializer(0.))
        for i in range(len(layer_sizes)-1):
            if(i == 0):
                f = tf.transpose(f, perm=[0, 1, 3, 2])
                f = layers.fully_connected(
                    f, layer_sizes[i+1]*2,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=normalizer_params)
                f = layers.dropout(f, drop_rate, is_training=True)
                f = tf.transpose(f, perm=[0, 1, 3, 2])
            elif(0 < i < len(layer_sizes)-2):
                w_mu = tf.zeros([1, layer_sizes[i+1], layer_sizes[i]])
                w = zs.Normal('w'+str(i), w_mu, std=1.,
                            n_samples=n_samples, group_ndims=2)
                w = tf.tile(w, [1, tf.shape(X)[0], 1, 2])
                f = tf.matmul(w, f) / tf.sqrt(layer_sizes[i]*1.)
                f = tf.concat([tf.cos(f), tf.sin(f)], 2)/\
                    (tf.exp(kern_logscale[i-1])*tf.sqrt(layer_sizes[i]*1.) )
            else:
                w_mu = tf.zeros([1, layer_sizes[i+1], layer_sizes[i]*2+1])
                w = zs.Normal('w'+str(i), w_mu, std=1.,
                            n_samples=n_samples, group_ndims=2)
                w = tf.tile(w, [1, tf.shape(X)[0], 1, 1])            
                f = tf.concat([f, tf.ones([n_samples, tf.shape(X)[0], 1, 1])], 2)
                f = tf.matmul(w, f)/tf.sqrt(layer_sizes[i]*2+1.)

        y_mean = tf.squeeze(f, [2, 3])
        
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        # y = zs.Laplace('y', y_mean, scale=tf.exp(y_logstd))
        y = zs.Normal('y', y_mean, logstd=y_logstd)

    return model, y_mean

@zs.reuse('variational')
def mean_field_variational(layer_sizes, drop_rate, n_samples):
    with zs.BayesianNet() as variational:
        for i in range(len(layer_sizes)-1):
            if(i == 0):
                pass
            elif(0 < i < len(layer_sizes)-2):
                w_mean = tf.get_variable('w_mean_'+str(i),
                    shape=[1, layer_sizes[i+1], layer_sizes[i]],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable('w_logstd_'+str(i),
                    shape=[1, layer_sizes[i+1], layer_sizes[i]],
                    initializer=tf.constant_initializer(0.))
                w = zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
            else:
                w_mean = tf.get_variable('w_mean_'+str(i),
                    shape=[1, layer_sizes[i+1], layer_sizes[i]*2+1],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable('w_logstd_'+str(i),
                    shape=[1, layer_sizes[i+1], layer_sizes[i]*2+1],
                    initializer=tf.constant_initializer(0.))
                w = zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
    return variational


def standardize(data_train, data_test):
    """
    Standardize a dataset to have zero mean and unit standard deviation.

    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.

    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std


if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1234)
    
    # Define model parameters
    drop_rate = 0.3
    n_hiddens = [50]

    # Define training/evaluation parameters
    plot_performance = True
    lb_samples = 20
    ll_samples = 500
    iters = 10
    epochs = 2000
    check_freq = 5
    early_stop = 30
    learning_rate = 1e-3

    train_test_set = load_data()
    train_lbs, train_rmses, train_lls = [], [], []
    test_lbs, test_rmses, test_lls = [], [], []
    for X_train, y_train, X_test, y_test in train_test_set:
        N, D = X_train.shape
        batch_size = int(np.floor(X_train.shape[0] / float(iters)))
    
        # Standardize data
        X_train, X_test, _, _ = standardize(X_train, X_test)
        y_train, y_test, mean_y_train, std_y_train = standardize(y_train, y_test)
    
    
        # Build the computation graph
        n_samples = tf.placeholder(tf.int32, shape=[], name='n_samples')
        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        X = tf.placeholder(tf.float32, shape=[None, D])
        y = tf.placeholder(tf.float32, shape=[None])
        y_obs = tf.tile(tf.expand_dims(y, 0), [n_samples, 1])
        layer_sizes = [D]+n_hiddens+[1]
        w_names = ['w' + str(i) for i in range(1, len(layer_sizes) - 1)]
    
        def log_joint(observed):
            model, _ = variational_activation_functions_neural_networks(
                observed, X, D, layer_sizes, drop_rate, n_samples, is_training)
            log_pws = model.local_log_prob(w_names)
            log_py_xw = model.local_log_prob('y')
            return tf.add_n(log_pws) + log_py_xw * N
    
        variational = mean_field_variational(layer_sizes, drop_rate, n_samples)
        qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
        latent = dict(zip(w_names, qw_outputs))
        
        lower_bound = zs.variational.elbo(
            log_joint, observed={'y': y_obs}, latent=latent, axis=0)
        cost = tf.reduce_mean(lower_bound.sgvb())
        lower_bound = tf.reduce_mean(lower_bound)
    
        learning_rate_ph = tf.placeholder(tf.float32, shape=[])
        global_step = tf.Variable(0, trainable=False)
        learning_rate_ts = tf.train.exponential_decay(
            learning_rate_ph, global_step, 10000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate_ts)
        infer_op = optimizer.minimize(cost, global_step=global_step)
        
    
        # prediction: rmse & log likelihood
        observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
        observed.update({'y': y_obs})
        model, y_mean = variational_activation_functions_neural_networks(
            observed, X, D, layer_sizes, drop_rate, n_samples, is_training)
        y_pred = tf.reduce_mean(y_mean, 0)
        rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
        log_py_xw = model.local_log_prob('y')
        log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
            tf.log(std_y_train)
    
        params = tf.trainable_variables()
        for i in params:
            print(i.name, i.get_shape())
    
        # Run the inference
        lb_window, count_over_train = [], 0
        fold_train_lbs, fold_train_rmses, fold_train_lls = [], [], []
        fold_test_lbs, fold_test_rmses, fold_test_lls = [], [], []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, epochs + 1):
                time_epoch = -time.time()
                lbs = []
                for t in range(iters):
                    X_batch = X_train[t * batch_size:(t + 1) * batch_size]
                    y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                    _, lb = sess.run(
                        [infer_op, lower_bound],
                        feed_dict={n_samples: lb_samples,
                                is_training: True,
                                learning_rate_ph: learning_rate,
                                X: X_batch, y: y_batch})
                    lbs.append(lb)
                time_epoch += time.time()
                print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                    epoch, time_epoch, np.mean(lbs)))
                if(len(lb_window)>=early_stop):
                    del lb_window[0]
                    if(np.mean(lbs)<np.mean(lb_window)):
                        count_over_train += 1
                    else:
                        count_over_train = 0
                lb_window.append(np.mean(lbs))
                if epoch % check_freq == 0:
                    time_train = -time.time()
                    train_lb, train_rmse, train_ll = sess.run(
                        [lower_bound, rmse, log_likelihood],
                        feed_dict={n_samples: lb_samples,
                                is_training: True,
                                X: X_train, y: y_train})
                    time_train += time.time()
                    if(len(fold_train_lbs)==0):
                        fold_train_lbs.append(train_lb)
                        fold_train_rmses.append(train_rmse)
                        fold_train_lls.append(train_ll)
                    else:
                        fold_train_lbs.append(train_lb)
                        fold_train_rmses.append(train_rmse)
                        fold_train_lls.append(train_ll)
                    print('>>> TRAIN ({:.1f}s)'.format(time_train))
                    print('>> Train lower bound = {}'.format(train_lb))
                    print('>> Train rmse = {}'.format(train_rmse))
                    print('>> Train log_likelihood = {}'.format(train_ll))
                    time_test = -time.time()
                    test_lb, test_rmse, test_ll = sess.run(
                        [lower_bound, rmse, log_likelihood],
                        feed_dict={n_samples: ll_samples,
                                is_training: False,
                                X: X_test, y: y_test})
                    time_test += time.time()
                    if(len(fold_test_lbs)==0):
                        fold_test_lbs.append(test_lb)
                        fold_test_rmses.append(test_rmse)
                        fold_test_lls.append(test_ll)
                    else:
                        fold_test_lbs.append(test_lb)
                        fold_test_rmses.append(test_rmse)
                        fold_test_lls.append(test_ll)
                    print('>>> TEST ({:.1f}s)'.format(time_test))
                    print('>> Test lower bound = {}'.format(test_lb))
                    print('>> Test rmse = {}'.format(test_rmse))
                    print('>> Test log_likelihood = {}'.format(test_ll))
                if(count_over_train > early_stop):
                    break
        train_lbs.append(np.array(fold_train_lbs))
        train_rmses.append(np.array(fold_train_rmses))
        train_lls.append(np.array(fold_train_lls))
        test_lbs.append(np.array(fold_test_lbs))
        test_rmses.append(np.array(fold_test_rmses))
        test_lls.append(np.array(fold_test_lls))
    print('>>> OVERALL TEST')
    print('>> Test lower bound = {}'.format(np.mean([lbs[-1] for lbs in test_lbs])))
    print('>> Test rmse = {}'.format(np.mean([lbs[-1] for lbs in test_rmses])))
    print('>> Test log_likelihood = {}'.format(np.mean([lbs[-1] for lbs in test_lls])))
    train_lbs = np.asarray(train_lbs)
    train_rmses = np.asarray(train_rmses)
    train_lls = np.asarray(train_lls)
    test_lbs = np.asarray(test_lbs)
    test_rmses = np.asarray(test_rmses)
    test_lls = np.asarray(test_lls)
    if(plot_performance):
        import matplotlib.pyplot as plt
        model_name = "VAFNN{"+",".join(list(map(str, layer_sizes)))+"}"
        plt.subplot(2, 1, 1)        
        plt.title(model_name+" on KEGG")
        test_epochs = (np.arange(len(train_rmses[0]))+1)*check_freq
        plt.semilogx(test_epochs, np.mean(train_rmses, axis=0), '--', label='Train')
        plt.semilogx(test_epochs, np.mean(test_rmses, axis=0), label='Test')
        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        
        plt.subplot(2, 1, 2)
        test_epochs = (np.arange(len(train_lls[0]))+1)*check_freq
        plt.semilogx(test_epochs, np.mean(train_lls, axis=0), '--', label='Train')
        plt.semilogx(test_epochs, np.mean(test_lls, axis=0), label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Log Likelihood')
        plt.show()
        plt.savefig(model_name+'_KEGG.png')