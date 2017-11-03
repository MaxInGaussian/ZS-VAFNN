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
    w_names = ['omega'+str(i) for i in range(len(net_sizes)-2)]
    return w_names

@zs.reuse('model')
def p_Y_Xw(observed, X, drop_rate, n_basis, net_sizes, n_samples, task):
    with zs.BayesianNet(observed=observed) as model:
        f = tf.expand_dims(tf.tile(X, [n_samples, 1, 1]), 1)
        omega = zs.Normal('temp'+str(i), M, std=1.,
                    n_samples=n_samples, group_ndims=2)
        for i in range(len(net_sizes)-1):
            f = tf.layers.dense(f, net_sizes[i+1],
                kernel_regularizer=layers.l1_l2_regularizer(1e-2, 1e-2),
                bias_regularizer=layers.l1_l2_regularizer(1e-2, 1e-2))
            if(i < len(net_sizes)-2):
                f = tf.nn.relu(f)
        f = tf.squeeze(f, [1])
        if(task == "classification"):
            f = tf.nn.softmax(f)
    return model, f, None

@zs.reuse('variational')
def var_q_w(n_basis, net_sizes, n_samples):
    with zs.BayesianNet() as variational:
        temp_mean = tf.get_variable('temp_mean'+str(i),
            shape=[], initializer=tf.constant_initializer(0.))
        temp_logstd = tf.get_variable('omega_logstd'+str(i),
            shape=[], initializer=tf.constant_initializer(0.))
        temp = zs.Normal('temp', temp_mean,
            logstd=temp_logstd, n_samples=n_samples, group_ndims=2)
    return variational
