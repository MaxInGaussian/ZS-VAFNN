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


def get_w_names(drop_rate, net_sizes):
    return ['w'+str(i) for i in range(len(net_sizes)-1)]

@zs.reuse('model')
def p_Y_Xw(observed, X, drop_rate, n_basis, net_sizes, n_samples, task):
    with zs.BayesianNet(observed=observed) as model:
        f = tf.expand_dims(tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1]), 2)
        for i in range(len(net_sizes)-1):
            w_mu = tf.zeros([1, net_sizes[i]+1, net_sizes[i+1]])
            w = zs.Normal('w'+str(i), w_mu, std=1.,
                          n_samples=n_samples, group_ndims=2)
            w = tf.tile(w, [1, tf.shape(X)[0], 1, 1])
            f = tf.concat([f, tf.ones([n_samples, tf.shape(X)[0], 1, 1])], 3)
            f = tf.matmul(f, w) / tf.sqrt(net_sizes[i]+1.)
            if(i < len(net_sizes)-2):
                f = tf.nn.relu(f)
        f = tf.squeeze(f, [2])
        if(task == "regression"):
            y_logstd = tf.get_variable('y_logstd', shape=[],
                                    initializer=tf.constant_initializer(0.))
            y = zs.Normal('y', f, logstd=y_logstd, group_ndims=1)
        elif(task == "classification"):
            y = zs.OnehotCategorical('y', f)
    return model, f, None

@zs.reuse('variational')
def var_q_w(n_basis, net_sizes, n_samples):
    with zs.BayesianNet() as variational:
        for i in range(len(net_sizes)-1):
            w_mean = tf.get_variable('w_mean'+str(i),
                shape=[1, net_sizes[i]+1, net_sizes[i+1]],
                initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable('w_logstd'+str(i),
                shape=[1, net_sizes[i]+1, net_sizes[i+1]],
                initializer=tf.constant_initializer(0.))
            zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                      n_samples=n_samples, group_ndims=2,
                      use_path_derivative=True)
    return variational