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
tf.set_random_seed(314159)

def get_w_names(drop_rate, net_sizes):
    w_names = []
    return w_names

@zs.reuse('model')
def p_Y_Xw(observed, X, drop_rate, n_basis, net_sizes, n_samples, task):
    with zs.BayesianNet(observed=observed) as model:
        f = tf.expand_dims(tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1]), 2)
        for i in range(len(net_sizes)-1):
            f = tf.layers.dense(f, net_sizes[i+1])
            if(i < len(net_sizes)-2):
                omega_mean = tf.get_variable('omega_mean'+str(i),
                    shape=[1, 1, net_sizes[i+1], n_basis],
                    initializer=tf.random_normal_initializer())
                omega_logstd = tf.get_variable('omega_logstd'+str(i),
                    shape=[1, 1, net_sizes[i+1], n_basis],
                    initializer=tf.random_normal_initializer())
                omega = omega_mean+tf.random_normal([
                    n_samples, 1, net_sizes[i+1], n_basis])*tf.exp(omega_logstd)
                omega = tf.tile(omega, [1, tf.shape(X)[0], 1, 1])
                f = tf.matmul(f, omega)/tf.sqrt(net_sizes[i+1]*1.)
                f = tf.concat([tf.cos(f), tf.sin(f)], 3)/tf.sqrt(n_basis*1.)
        f = tf.squeeze(f, [2])
        if(task == "regression"):
            y_logstd = tf.get_variable('y_logstd', shape=[],
                                    initializer=tf.constant_initializer(0.))
            y = zs.Normal('y', f, logstd=y_logstd, group_ndims=1)
        elif(task == "classification"):
            y = zs.OnehotCategorical('y', f)
    return model, f, None
