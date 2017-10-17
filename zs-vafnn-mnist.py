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
def variational_activation_functions_neural_networks(
    observed, x, n_x, layer_sizes, n_samples, is_training):
    with zs.BayesianNet(observed=observed) as model:
        f = tf.expand_dims(tf.tile(tf.expand_dims(x, 0), [n_samples, 1, 1]), 3)
        kern_logscale = tf.get_variable('kern_logscale',
            shape=[len(layer_sizes)-1], initializer=tf.constant_initializer(0.))

        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        for i in range(len(layer_sizes)-1):
            
            if(i < len(layer_sizes)-2):
                
                f = tf.transpose(f, perm=[0, 1, 3, 2])
                f = layers.fully_connected(
                    f, layer_sizes[i], activation_fn=None,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=normalizer_params)
                f = tf.transpose(f, perm=[0, 1, 3, 2])
                
                f = tf.concat([tf.cos(f), tf.sin(f)], 2)/tf.exp(kern_logscale[i])/tf.sqrt(layer_sizes[i]*1.) 
                w_mu = tf.zeros([1, layer_sizes[i+1], layer_sizes[i]])
                w = zs.Normal('w'+str(i), w_mu, std=2.*np.pi,
                            n_samples=n_samples, group_ndims=2)
                w = tf.tile(w, [1, tf.shape(x)[0], 1, 2])

                f = tf.matmul(w, f)
                
            else:
                w_mu = tf.zeros([1, layer_sizes[i+1], layer_sizes[i]+1])
                w = zs.Normal('w'+str(i), w_mu, std=1.,
                            n_samples=n_samples, group_ndims=2)
                w = tf.tile(w, [1, tf.shape(x)[0], 1, 1])
                
                f = tf.concat([f, tf.ones([n_samples, tf.shape(x)[0], 1, 1])], 2)
                
                f = tf.matmul(w, f) / tf.sqrt(layer_sizes[i]+1.)

        f = tf.squeeze(f, [3])

        y = zs.OnehotCategorical('y', f)

    return model, y


def mean_field_variational(layer_sizes, n_samples):
    with zs.BayesianNet() as variational:
        for i in range(len(layer_sizes)-1):            
            if(i < len(layer_sizes)-2):
                w_mean = tf.get_variable('w_mean_'+str(i),
                    shape=[1, layer_sizes[i+1], layer_sizes[i]],
                    initializer=tf.constant_initializer(np.log(2.*np.pi)))
                w_logstd = tf.get_variable('w_logstd_'+str(i),
                shape=[1, layer_sizes[i+1], layer_sizes[i]],
                    initializer=tf.constant_initializer(np.log(2.*np.pi)))
                zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
            else:
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
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.vstack([y_train, y_valid])
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    n_x = x_train.shape[1]
    n_class = 10

    # Define training/evaluation parameters
    epochs = 500
    batch_size = 1000
    lb_samples = 10
    ll_samples = 100
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 3
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75
    
    # placeholders
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    y = tf.placeholder(tf.int32, shape=(None, n_class))
    n = tf.shape(x)[0]

    net_size = [n_x, 100, 100, 100, n_class]
    e_names = ['w' + str(i) for i in range(len(net_size) - 1)]

    def log_joint(observed):
        model, _ = variational_activation_functions_neural_networks(
            observed, x, n, net_size, n_particles, is_training)
        log_pe = model.local_log_prob(e_names)
        log_py_xe = model.local_log_prob('y')
        return tf.add_n(log_pe) / x_train.shape[0] + log_py_xe

    variational = mean_field_variational(net_size, n_particles)
    qe_queries = variational.query(e_names, outputs=True, local_log_prob=True)
    qe_samples, log_qes = zip(*qe_queries)
    log_qes = [log_qe / x_train.shape[0] for log_qe in log_qes]
    e_dict = dict(zip(e_names, zip(qe_samples, log_qes)))
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'y': y}, e_dict, axis=0))

    _, h_pred = variational_activation_functions_neural_networks(
        dict(zip(e_names, qe_samples)), x, n, net_size, n_particles, is_training)
    h_pred = tf.reduce_mean(h_pred, 0)
    y_pred = tf.argmax(h_pred, 1)
    sparse_y = tf.argmax(y, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, sparse_y), tf.float32))

    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer = optimizer.minimize(-lower_bound)

    params = tf.trainable_variables()
    for i in params:
        print('variable name = {}, shape = {}'
            .format(i.name, i.get_shape()))

    # Run the inference
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            time_epoch = -time.time()
            indices = np.random.permutation(x_train.shape[0])
            x_train = x_train[indices]
            y_train = y_train[indices]
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
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
                test_lbs = []
                test_accs = []
                for t in range(10):
                    x_batch = x_test[t * 1000:(t + 1) * 1000]
                    y_batch = y_test[t * 1000:(t + 1) * 1000]
                    lb, acc1 = sess.run(
                        [lower_bound, acc],
                        feed_dict={n_particles: ll_samples,
                                   is_training: False,
                                   x: x_batch, y: y_batch})
                    test_lbs.append(lb)
                    test_accs.append(acc1)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test accuaracy = {}'.format(np.mean(test_accs)))
