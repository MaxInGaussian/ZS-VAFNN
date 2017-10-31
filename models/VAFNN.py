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

import numpy as np
import zhusuan as zs
import tensorflow as tf
import tensorflow.contrib.layers as layers


def get_w_names(drop_rate, net_sizes):
    w_names = ['w'+str(i) for i in range(len(net_sizes)-1)]
    if(1-drop_rate > 1e-3):
        w_names += ['omega'+str(i) for i in range(len(net_sizes)-2)]
    return w_names

@zs.reuse('model')
def p_Y_Xw(observed, X, drop_rate, n_basis, net_sizes, n_samples, task):
    with zs.BayesianNet(observed=observed) as model:
        f = tf.expand_dims(tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1]), 2)
        KL_V = 0
        for i in range(len(net_sizes)-1):
            if(i < len(net_sizes)-2):
                if(i == 0):
                    A = tf.get_variable('A'+str(i),
                        shape=[1, 1, net_sizes[i], net_sizes[i+1]],
                        initializer=tf.constant_initializer(0.))
                    A = tf.tile(A, [n_samples, tf.shape(X)[0], 1, 1])
                else:
                    A = tf.get_variable('A'+str(i),
                        shape=[1, 1, n_basis, net_sizes[i+1]],
                        initializer=tf.constant_initializer(0.))
                    A = tf.tile(A, [n_samples, tf.shape(X)[0], 2, 1])
                b = tf.get_variable('b'+str(i),
                    shape=[1, 1, 1, net_sizes[i+1]],
                    initializer=tf.constant_initializer(0.))
                b = tf.tile(b, [n_samples, tf.shape(X)[0], 1, 1])
                f = (tf.matmul(f, A)+b)/tf.sqrt(net_sizes[i]*1.)
                # f = layers.dropout(f, drop_rate)
                w_p = tf.zeros([1, 1, net_sizes[i+1]])+drop_rate
                w = zs.Bernoulli('w'+str(i), w_p,
                    n_samples=n_samples, group_ndims=2)
                f = f*tf.cast(w, tf.float32)
                V = tf.get_variable('omega_var'+str(i),
                    shape=[1, 1, net_sizes[i+1], n_basis],
                    initializer=tf.constant_initializer(0.))
                V = tf.tile(tf.abs(V), [n_samples, tf.shape(X)[0], 1, 1])
                expVf2 = tf.exp(-2*np.pi**2*tf.matmul(f**2, V))
                M = tf.get_variable('omega_mean'+str(i),
                    shape=[1, net_sizes[i+1], n_basis],
                    initializer=tf.constant_initializer(0.))
                omega = 2*np.pi*zs.Normal('omega'+str(i), M, std=1.,
                            n_samples=n_samples, group_ndims=2)
                omega = tf.tile(omega, [1, tf.shape(X)[0], 1, 1])
                f = tf.matmul(f, omega)/tf.sqrt(net_sizes[i+1]*1.)
                f = tf.concat([expVf2*tf.cos(f), expVf2*tf.sin(f)], 3)
                KL_V += tf.reduce_mean(M**2+V-tf.log(V+1e-8))
                continue
            w_mu = tf.zeros([1, n_basis*2+1, net_sizes[i+1]])
            w = zs.Normal('w'+str(i), w_mu, std=1.,
                        n_samples=n_samples, group_ndims=2)
            w = tf.tile(w, [1, tf.shape(X)[0], 1, 1])
            f = tf.concat([f, tf.ones([n_samples, tf.shape(X)[0], 1, 1])], 3)
            f = tf.matmul(f, w)/tf.sqrt(net_sizes[i]*1.)
        f = tf.squeeze(f, [2])
        if(task == "regression"):
            y_logstd = tf.get_variable('y_logstd', shape=[],
                                    initializer=tf.constant_initializer(0.))
            y = zs.Normal('y', f, logstd=y_logstd, group_ndims=1)
        elif(task == "classification"):
            y = zs.OnehotCategorical('y', f)
    return model, f, KL_V

@zs.reuse('variational')
def var_q_w(n_basis, net_sizes, n_samples):
    with zs.BayesianNet() as variational:
        for i in range(len(net_sizes)-1):
            if(i < len(net_sizes)-2):
                w_p = tf.get_variable('w_p'+str(i), shape=[1, 1, net_sizes[i+1]],
                    initializer=tf.constant_initializer(0.5))
                zs.Bernoulli('w'+str(i), w_p, n_samples=n_samples, group_ndims=2)
                omega_mean = tf.get_variable('omega_mean'+str(i),
                    shape=[1, net_sizes[i+1], n_basis],
                    initializer=tf.constant_initializer(0.))
                omega_logstd = tf.get_variable('omega_logstd'+str(i),
                    shape=[1, net_sizes[i+1], n_basis],
                    initializer=tf.constant_initializer(0.))
                omega = zs.Normal('omega'+str(i), omega_mean,
                    logstd=omega_logstd, n_samples=n_samples, group_ndims=2)
                pass
            else:
                w_mean = tf.get_variable('w_mean'+str(i),
                    shape=[1, n_basis*2+1, net_sizes[i+1]],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable('w_logstd'+str(i),
                    shape=[1, n_basis*2+1, net_sizes[i+1]],
                    initializer=tf.constant_initializer(0.))
                w = zs.Normal('w'+str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
                pass
    return variational
