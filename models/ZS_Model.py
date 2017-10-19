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

class DeepModel(object):

    def __init__(self, hidden_layers, **args):
        self.layer_sizes = [-1]+n_hiddens+[-1]
    
    def build_computation_graph(self, X_train, Y_train):
        
        (N, D), P = X_train.shape, Y_train.shape[1]
        layer_sizes[0], layer_sizes[-1] = D, P
        
        n_samples = tf.placeholder(tf.int32, shape=[], name='n_samples')
        X = tf.placeholder(tf.float32, shape=[None, D])
        Y = tf.placeholder(tf.float32, shape=[None, P])
        Y_obs = tf.tile(tf.expand_dims(Y, 0), [n_samples, 1])
        layer_sizes = [D] + n_hiddens + [P]
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