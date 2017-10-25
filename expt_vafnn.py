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

import os
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from six.moves import range, zip
import numpy as np
import zhusuan as zs

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
            # if(i == 0):
            #     f = tf.transpose(f, perm=[0, 1, 3, 2])
            #     f = layers.fully_connected(
            #         f, layer_sizes[i+1],
            #         normalizer_fn=layers.batch_norm,
            #         normalizer_params=normalizer_params)
            #     f = layers.dropout(f, drop_rate, is_training=True)
            #     f = tf.transpose(f, perm=[0, 1, 3, 2])
            if(i < len(layer_sizes)-2):
                w_mu = tf.zeros([1, layer_sizes[i+1]//2, layer_sizes[i]+1])
                w = zs.Normal('w'+str(i), w_mu, std=1.,
                            n_samples=n_samples, group_ndims=2)
                w = tf.tile(w, [1, tf.shape(X)[0], 2, 1])
                f = tf.concat([f, tf.ones([n_samples, tf.shape(X)[0], 1, 1])], 2)
                f = tf.matmul(w, f) / tf.sqrt(layer_sizes[i]*1.)
                f = tf.concat([tf.cos(f), tf.sin(f)], 2)/\
                    (tf.exp(kern_logscale[i-1])*tf.sqrt(layer_sizes[i]*1.) )
            else:
                w_mu = tf.zeros([1, layer_sizes[i+1], layer_sizes[i]])
                w = zs.Normal('w'+str(i), w_mu, std=1.,
                            n_samples=n_samples, group_ndims=2)
                w = tf.tile(w, [1, tf.shape(X)[0], 1, 2])
                f = tf.matmul(w, f)/tf.sqrt(layer_sizes[i]*1.)
        y_mean = tf.squeeze(f, [2, 3])
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Laplace('y', y_mean, scale=tf.exp(y_logstd))
    return model, y_mean

@zs.reuse('variational')
def mean_field_variational(layer_sizes, n_samples):
    with zs.BayesianNet() as variational:
        for i in range(len(layer_sizes)-1):
            if(i < len(layer_sizes)-2):
                w_mean = tf.get_variable('w_mean_'+str(i),
                    shape=[1, layer_sizes[i+1]//2, layer_sizes[i]+1],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable('w_logstd_'+str(i),
                    shape=[1, layer_sizes[i+1]//2, layer_sizes[i]+1],
                    initializer=tf.constant_initializer(0.))
                w = zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
            else:
                w_mean = tf.get_variable('w_mean_'+str(i),
                    shape=[1, layer_sizes[i+1], layer_sizes[i]],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable('w_logstd_'+str(i),
                    shape=[1, layer_sizes[i+1], layer_sizes[i]],
                    initializer=tf.constant_initializer(0.))
                w = zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
    return variational


def standardize(data_train, data_test):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std


def run_vafnn_experiment(dataset_name, train_test_set, **args):
    
    tf.set_random_seed(1237)
    np.random.seed(1234)
    
    # Define model parameters
    drop_rate = 0.3 if 'drop_rate' not in args.keys() else args['drop_rate']
    n_hiddens = [50] if 'n_hiddens' not in args.keys() else args['n_hiddens']

    # Define training/evaluation parameters
    plot_err = True if 'plot_err' not in args.keys() else args['plot_err']
    lb_samples = 20 if 'lb_samples' not in args.keys() else args['lb_samples']
    ll_samples = 100 if 'll_samples' not in args.keys() else args['ll_samples']
    batch_size = 50 if 'batch_size' not in args.keys() else args['batch_size']
    max_epochs = 2000 if 'max_epochs' not in args.keys() else args['max_epochs']
    check_freq = 5 if 'check_freq' not in args.keys() else args['check_freq']
    early_stop = 20 if 'early_stop' not in args.keys() else args['early_stop']
    learn_rate = 1e-3 if 'learn_rate' not in args.keys() else args['learn_rate']
    
    eval_rmses, eval_lls = [], []
    train_lbs, train_rmses, train_lls = [], [], []
    test_lbs, test_rmses, test_lls = [], [], []
    for fold, (X_train, y_train, X_test, y_test) in enumerate(train_test_set):
        
        problem_name = dataset_name.replace(' ', '_')+'_'+str(fold+1)
        N, D = X_train.shape
        iters = int(np.floor(X_train.shape[0] / float(batch_size)))
    
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
        w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]
        model_name = "VAFNN{"+",".join(list(map(str, layer_sizes)))+"}"
    
        def log_joint(observed):
            model, _ = variational_activation_functions_neural_networks(
                observed, X, D, layer_sizes, drop_rate, n_samples, is_training)
            log_pws = model.local_log_prob(w_names)
            log_py_xw = model.local_log_prob('y')
            return tf.add_n(log_pws) + log_py_xw * N
    
        variational = mean_field_variational(layer_sizes, n_samples)
        qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
        latent = dict(zip(w_names, qw_outputs))
        
        lower_bound = zs.variational.elbo(
            log_joint, observed={'y': y_obs}, latent=latent, axis=0)
        cost = tf.reduce_mean(lower_bound.sgvb())
        lower_bound = tf.reduce_mean(lower_bound)
    
        learn_rate_ph = tf.placeholder(tf.float32, shape=[])
        global_step = tf.Variable(0, trainable=False)
        learn_rate_ts = tf.train.exponential_decay(
            learn_rate_ph, global_step, 10000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learn_rate_ts)
        infer_op = optimizer.minimize(cost, global_step=global_step)
        
    
        # prediction: rmse & log likelihood
        observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
        observed.update({'y': y_obs})
        model, y_mean = variational_activation_functions_neural_networks(
            observed, X, D, layer_sizes, drop_rate, n_samples, is_training)
        y_pred = tf.reduce_mean(y_mean, 0)
        rms_error = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
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
            max_lb, min_err = -np.Infinity, np.Infinity
            for epoch in range(1, max_epochs + 1):
                time_epoch = -time.time()
                lbs = []
                for t in range(iters):
                    if(t == iters-1):                        
                        X_batch = X_train[t*batch_size:]
                        y_batch = y_train[t*batch_size:]
                    else:
                        X_batch = X_train[t*batch_size:(t+1)*batch_size]
                        y_batch = y_train[t*batch_size:(t+1)*batch_size]
                    _, lb = sess.run(
                        [infer_op, lower_bound],
                        feed_dict={n_samples: lb_samples,
                                is_training: True,
                                learn_rate_ph: learn_rate,
                                X: X_batch, y: y_batch})
                    lbs.append(lb)
                time_epoch += time.time()
                print('Epoch {} ({:.1f}s, {}): Lower bound = {}'.format(
                    epoch, time_epoch, count_over_train, np.mean(lbs)))
                if(len(lb_window)>=early_stop):
                    del lb_window[0]
                    if(max_lb>np.max(lb_window)+2*np.std(lb_window)):
                        count_over_train += 1
                lb_window.append(np.mean(lbs))
                if(max_lb < np.mean(lbs)):
                    max_lb = np.mean(lbs)
                if epoch % check_freq == 0:
                    lbs, rmses, lls = [], [], []
                    time_train = -time.time()
                    for t in range(iters):
                        if(t == iters-1):                        
                            X_batch = X_train[t*batch_size:]
                            y_batch = y_train[t*batch_size:]
                        else:
                            X_batch = X_train[t*batch_size:(t+1)*batch_size]
                            y_batch = y_train[t*batch_size:(t+1)*batch_size]
                        lb, rmse, ll = sess.run(
                            [lower_bound, rms_error, log_likelihood],
                            feed_dict={n_samples: ll_samples,
                                    is_training: False,
                                    X: X_batch, y: y_batch})
                        lbs.append(lb);rmses.append(rmse);lls.append(ll)
                    train_lb, train_rmse, train_ll =\
                        np.mean(lbs), np.mean(rmses), np.mean(lls)
                    time_train += time.time()
                    if(min_err>train_rmse/np.exp(train_ll)):
                        count_over_train = 0
                        min_err = train_rmse/np.exp(train_ll)
                        saver = tf.train.Saver()
                        if not os.path.exists('./trained/'):
                            os.makedirs('./trained/')
                        saver.save(sess, './trained/'+model_name+'_'+problem_name)
                    else:
                        count_over_train += 1
                    if(len(fold_train_lbs)==0):
                        fold_train_lbs.append(train_lb)
                        fold_train_rmses.append(train_rmse)
                        fold_train_lls.append(train_ll)
                    else:
                        fold_train_lbs.append(train_lb)
                        fold_train_rmses.append(train_rmse)
                        fold_train_lls.append(train_ll)
                    print('>>> TRAIN ({:.1f}s) - min_err = {}'.format(time_train, min_err))
                    print('>> Train lower bound = {}'.format(train_lb))
                    print('>> Train rmse = {}'.format(train_rmse))
                    print('>> Train log_likelihood = {}'.format(train_ll))
                    lbs, rmses, lls = [], [], []
                    time_test = -time.time()
                    t_iters = int(np.floor(X_test.shape[0] / float(batch_size)))
                    for t in range(t_iters):
                        if(t == t_iters-1):                   
                            X_batch = X_test[t*batch_size:]
                            y_batch = y_test[t*batch_size:]
                        else:
                            X_batch = X_test[t*batch_size:(t+1)*batch_size]
                            y_batch = y_test[t*batch_size:(t+1)*batch_size]
                        lb, rmse, ll = sess.run(
                            [lower_bound, rms_error, log_likelihood],
                            feed_dict={n_samples: ll_samples,
                                    is_training: False,
                                    X: X_batch, y: y_batch})
                        lbs.append(lb);rmses.append(rmse);lls.append(ll)
                    test_lb, test_rmse, test_ll =\
                        np.mean(lbs), np.mean(rmses), np.mean(lls)
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
            
            # Load the selected best params and evaluate its performance
            saver = tf.train.Saver()
            saver.restore(sess, './trained/'+model_name+'_'+problem_name)
            lbs, rmses, lls = [], [], []
            t_iters = int(np.floor(X_test.shape[0] / float(batch_size)))
            for t in range(t_iters):
                if(t == t_iters-1):                   
                    X_batch = X_test[t*batch_size:]
                    y_batch = y_test[t*batch_size:]
                else:
                    X_batch = X_test[t*batch_size:(t+1)*batch_size]
                    y_batch = y_test[t*batch_size:(t+1)*batch_size]
                lb, rmse, ll = sess.run(
                    [lower_bound, rms_error, log_likelihood],
                    feed_dict={n_samples: ll_samples,
                            is_training: False,
                            X: X_batch, y: y_batch})
                lbs.append(lb);rmses.append(rmse);lls.append(ll)
            test_lb, test_rmse, test_ll =\
                np.mean(lbs), np.mean(rmses), np.mean(lls)
            print('>>> BEST TEST')
            print('>> Test lower bound = {}'.format(test_lb))
            print('>> Test rmse = {}'.format(test_rmse))
            print('>> Test log_likelihood = {}'.format(test_ll))
            eval_rmses.append(test_rmse)
            eval_lls.append(test_ll)
        if(plot_err):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 1, 1)        
            plt.title(model_name+" on "+dataset_name)
            test_max_epochs = (np.arange(len(fold_train_rmses))+1)*check_freq
            plt.semilogx(test_max_epochs, fold_train_rmses, '--', label='Train')
            plt.semilogx(test_max_epochs, fold_test_rmses, label='Test')
            plt.legend(loc='upper right')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            
            plt.subplot(2, 1, 2)
            test_max_epochs = (np.arange(len(fold_train_lls))+1)*check_freq
            plt.semilogx(test_max_epochs, fold_train_lls, '--', label='Train')
            plt.semilogx(test_max_epochs, fold_test_lls, label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Log Likelihood')
            if not os.path.exists('./plots/'):
                os.makedirs('./plots/')
            plt.savefig('./plots/'+model_name+'_'+problem_name+'.png')
            plt.close()
        train_lbs.append(np.array(fold_train_lbs))
        train_rmses.append(np.array(fold_train_rmses))
        train_lls.append(np.array(fold_train_lls))
        test_lbs.append(np.array(fold_test_lbs))
        test_rmses.append(np.array(fold_test_rmses))
        test_lls.append(np.array(fold_test_lls))
    print('>>> OVERALL TEST')
    print('>> Overall Test rmse = {}'.format(np.mean(eval_rmses)))
    print('>> Overall Test log_likelihood = {}'.format(np.mean(eval_lls)))
    return eval_rmses, eval_lls