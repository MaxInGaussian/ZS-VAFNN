#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def bayesian_neural_networks(observed, x, n_x, layer_sizes, n_samples):
    with zs.BayesianNet(observed=observed) as model:
        f = tf.expand_dims(tf.tile(tf.expand_dims(x, 0), [n_samples, 1, 1]), 3)
        # shape = {n_samples}*batch_size*n_x*1
        
        for i in range(len(layer_sizes)-1):
            w_mu = tf.zeros([1, layer_sizes[i+1], layer_sizes[i]+1])
            w = zs.Normal('w'+str(i), w_mu, std=1.,
                          n_samples=n_samples, group_ndims=2)
            w = tf.tile(w, [1, tf.shape(x)[0], 1, 1])
            # shape = {n_samples}*batch_size*layer_sizes[i+1]*(layer_sizes[i]+1)
            
            f = tf.concat([f, tf.ones([n_samples, tf.shape(x)[0], 1, 1])], 2)
            # shape = {n_samples}*batch_size*(layer_sizes[i]+1)*1
            
            f = tf.matmul(w, f) / tf.sqrt(layer_sizes[i]+1.)
            # shape = {n_samples}*batch_size*layer_sizes[i+1]*1
            
            if(i < len(layer_sizes)-2):
                # f = tf.concat([f, f])
                f = tf.nn.relu(f)

        y_mean = tf.squeeze(f, [2, 3])
        # shape = {n_samples}*batch_size
            
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Laplace('y', y_mean, scale=tf.exp(y_logstd))
        # y = zs.Normal('y', y_mean, logstd=y_logstd)

    return model, y_mean


def mean_field_variational(layer_sizes, n_samples):
    with zs.BayesianNet() as variational:
        for i in range(len(layer_sizes)-1):
            w_mean = tf.get_variable('w_mean_'+str(i),
                shape=[1, layer_sizes[i+1], layer_sizes[i]+1],
                initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable('w_logstd_'+str(i),
            shape=[1, layer_sizes[i+1], layer_sizes[i]+1],
                initializer=tf.constant_initializer(0.))
            zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                      n_samples=n_samples, group_ndims=2)
    return variational


if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, 'housing.data')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    N, n_x = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Define model parameters
    n_hiddens = [100, 50]

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 5000
    epochs = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10
    learning_rate = 1
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    # Build the computation graph
    n_samples = tf.placeholder(tf.int32, shape=[], name='n_samples')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_samples, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_joint(observed):
        model, _ = bayesian_neural_networks(
            observed, x, n_x, layer_sizes, n_samples)
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

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdadeltaOptimizer(learning_rate_ph)
    infer_op = optimizer.minimize(cost)

    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y_obs})
    model, y_mean = bayesian_neural_networks(
        observed, x, n_x, layer_sizes, n_samples)
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
        tf.log(std_y_train)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            # if epoch % anneal_lr_freq == 0:
            #     learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer_op, lower_bound],
                    feed_dict={n_samples: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lb, test_rmse, test_ll = sess.run(
                    [lower_bound, rmse, log_likelihood],
                    feed_dict={n_samples: ll_samples,
                               x: x_test, y: y_test})
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test rmse = {}'.format(test_rmse))
                print('>> Test log_likelihood = {}'.format(test_ll))