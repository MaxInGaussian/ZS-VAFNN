#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def VAFNN(observed, x, n_x, layer_sizes, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        h = tf.expand_dims(
                tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        h = tf.concat([h, h], 2)
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        omegas = []
        for i, (n_in,n_out) in enumerate(zip(layer_sizes[:-1],layer_sizes[1:])):
            omega_mu = tf.zeros([1, n_out, tf.shape(h)[2]])
            omegas.append(zs.Normal('omega' + str(i), omega_mu, std=1.,
                                n_samples=n_particles, group_ndims=2))
            omega = tf.tile(omegas[i], [1, tf.shape(x)[0], 1, 1])
            h = tf.matmul(omega, h)
            if(i < len(layer_sizes)-2):
                h = tf.concat([tf.cos(h), tf.sin(h)], 2)/tf.sqrt(tf.cast(n_out,
                                                        tf.float32))
            h = layers.fully_connected(
                h, n_out, normalizer_fn=layers.batch_norm,
                normalizer_params=normalizer_params)
        
        y_mean = tf.squeeze(h, [2, 3])
        y_logscale = tf.get_variable('y_logscale', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Laplace('y', h, scale=tf.exp(y_logscale))

    return model, h


def mean_field_variational(layer_sizes, n_particles):
    with zs.BayesianNet() as variational:
        omegas = []
        for i, (n_in,n_out) in enumerate(zip(layer_sizes[:-1],layer_sizes[1:])):
            omega_mean = tf.get_variable(
                'omega_mean_' + str(i), shape=[1, n_out, n_in*2],
                initializer=tf.constant_initializer(0.))
            omega_logstd = tf.get_variable(
                'omega_logstd_' + str(i), shape=[1, n_out, n_in*2],
                initializer=tf.constant_initializer(0.))
            omegas.append(
                zs.Normal('omega' + str(i), omega_mean, logstd=omega_logstd,
                        n_samples=n_particles, group_ndims=2))
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
    n_hiddens = [25]

    # Define training/evaluation parameters
    lb_samples = 25
    ll_samples = 5000
    epochs = 300
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    omega_names = ['omega' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_joint(observed):
        model, _ = VAFNN(observed, x, n_x, layer_sizes, n_particles, is_training)
        log_pws = model.local_log_prob(omega_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + log_py_xw * N

    variational = mean_field_variational(layer_sizes, n_particles)
    qw_outputs = variational.query(omega_names, outputs=True, local_log_prob=True)
    latent = dict(zip(omega_names, qw_outputs))
    lower_bound = zs.variational.elbo(
        log_joint, observed={'y': y_obs}, latent=latent, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    infer_op = optimizer.minimize(cost)

    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in omega_names)
    observed.update({'y': y_obs})
    model, y_mean = VAFNN(observed, x, n_x, layer_sizes, n_particles, is_training)
    y_pred = tf.reduce_mean(y_mean, 0)
    mae = tf.reduce_mean(tf.abs(y_pred - y) * std_y_train)
    nmse = tf.reduce_mean((y_pred - y) ** 2)
    rmse = tf.sqrt(nmse) * std_y_train
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
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer_op, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               is_training: True,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lb, test_mae, test_nmse, test_rmse, test_ll = sess.run(
                    [lower_bound, mae, nmse, rmse, log_likelihood],
                    feed_dict={n_particles: ll_samples,
                               is_training: False,
                               x: x_test, y: y_test})
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test mae = {}'.format(test_mae))
                print('>> Test nmse = {}'.format(test_nmse))
                print('>> Test rmse = {}'.format(test_rmse))
                print('>> Test log_likelihood = {}'.format(test_ll))
