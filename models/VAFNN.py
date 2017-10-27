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


def get_w_names(net_sizes):
    return ['w'+str(i) for i in range(len(net_sizes)-1)]

@zs.reuse('model')
def p_Y_Xw(observed, X, n_basis, net_sizes, n_samples, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
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
                        shape=[1, 1, n_basis*2, net_sizes[i+1]],
                        initializer=tf.constant_initializer(0.))
                    A = tf.tile(A, [n_samples, tf.shape(X)[0], 1, 1])
                b = tf.get_variable('b'+str(i),
                    shape=[1, 1, 1, net_sizes[i+1]],
                    initializer=tf.constant_initializer(0.))
                b = tf.tile(b, [n_samples, tf.shape(X)[0], 1, 1])
                f = (tf.matmul(f, A)+b)/tf.sqrt(net_sizes[i]*1.)
                V = tf.get_variable('V'+str(i),
                    shape=[1, 1, net_sizes[i+1], n_basis],
                    initializer=tf.constant_initializer(0.))
                V = tf.tile(V, [n_samples, tf.shape(X)[0], 1, 1])
                expVf2 = tf.exp(-2*np.pi**2*tf.matmul(f**2, V))
                w_mu = tf.zeros([1, net_sizes[i+1], n_basis])
                w = 2*np.pi*zs.Normal('w'+str(i), w_mu, std=1.,
                            n_samples=n_samples, group_ndims=2)
                w = tf.tile(w, [1, tf.shape(X)[0], 1, 1])
                f = tf.matmul(f, w)/tf.sqrt(net_sizes[i+1]*1.)
                f = tf.concat([expVf2*tf.cos(f), expVf2*tf.sin(f)], 3)
                M, V = tf.reduce_mean(w, 0), tf.reduce_mean(V, 0)
                KL_V += tf.reduce_mean(V**2-tf.log(V**2+1e-8))
                continue
            w_mu = tf.zeros([1, n_basis*2, net_sizes[i+1]])
            w = zs.Normal('w'+str(i), w_mu, std=1.,
                        n_samples=n_samples, group_ndims=2)
            w = tf.tile(w, [1, tf.shape(X)[0], 1, 1])
            f = tf.matmul(f, w)/tf.sqrt(net_sizes[i]*1.)
        y_mean = tf.squeeze(f, [3])
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, logstd=y_logstd, group_ndims=1)
    return model, y_mean, KL_V

@zs.reuse('variational')
def var_q_w(n_basis, net_sizes, n_samples):
    with zs.BayesianNet() as variational:
        for i in range(len(net_sizes)-1):
            if(i < len(net_sizes)-2):
                w_mean = tf.get_variable('w_mean_'+str(i),
                    shape=[1, net_sizes[i+1], n_basis],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable('w_logstd_'+str(i),
                    shape=[1, net_sizes[i+1], n_basis],
                    initializer=tf.constant_initializer(0.))
                w = zs.Normal('w'+str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
                pass
            else:
                w_mean = tf.get_variable('w_mean_'+str(i),
                    shape=[1, n_basis*2, net_sizes[i+1]],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable('w_logstd_'+str(i),
                    shape=[1, n_basis*2, net_sizes[i+1]],
                    initializer=tf.constant_initializer(0.))
                w = zs.Normal('w'+str(i), w_mean, logstd=w_logstd,
                        n_samples=n_samples, group_ndims=2)
                pass
    return variational
