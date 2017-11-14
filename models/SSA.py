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
        KL_w = 0
        for i in range(len(net_sizes)-1):
            if(i < len(net_sizes)-2):
                f = tf.layers.dense(f, n_basis)
                alpha_mean = tf.get_variable('alpha_mean'+str(i),
                    shape=[1, 1, n_basis*2, net_sizes[i+1]],
                    initializer=tf.random_normal_initializer())
                alpha_logstd = tf.get_variable('alpha_logstd'+str(i),
                    shape=[1, 1, n_basis*2, net_sizes[i+1]],
                    initializer=tf.random_normal_initializer())
                alpha_std = tf.exp(alpha_logstd)
                alpha = alpha_mean+tf.random_normal([
                    n_samples, 1, n_basis*2, net_sizes[i+1]])*alpha_std
                alpha = tf.tile(alpha, [1, tf.shape(X)[0], 1, 1])
                f1 = tf.matmul(f, alpha[:,:,:n_basis,:])/tf.sqrt(n_basis*.5)
                f2 = tf.matmul(f, alpha[:,:,n_basis:,:])/tf.sqrt(n_basis*.5)
                f = tf.concat([tf.cos(f1)+tf.cos(f2),
                    tf.sin(f1)+tf.sin(f2)], 3)/tf.sqrt(net_sizes[i+1]*1.)
                KL_w += tf.reduce_mean(
                    alpha_std**2+alpha_mean**2-2*alpha_logstd-1)/2.
            else:
                f = tf.layers.dense(f, net_sizes[i+1])
        f = tf.squeeze(f, [2])
    return model, f, KL_w/(len(net_sizes)-2)
