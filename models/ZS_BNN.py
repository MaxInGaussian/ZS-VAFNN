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

from .ZS_Model import DeepModel

class BNN(DeepModel):

    def __init__(self, hidden_layers, activation=tf.nn.relu, **args):
        super(BNN, self).__init__(hidden_layers, **args)
        self.activation = activation
        
    def fit(self, X_train, Y_train):
        
        @zs.reuse('model')
        def bayesian_neural_networks(observed, x, n_samples):
            with zs.BayesianNet(observed=observed) as model:
                f = tf.expand_dims(tf.tile(tf.expand_dims(x, 0), [n_samples, 1, 1]), 3)
                for i in range(len(self.layer_sizes)-1):
                    w_mu = tf.zeros([1, self.layer_sizes[i+1], self.layer_sizes[i]+1])
                    w = zs.Normal('w'+str(i), w_mu, std=1.,
                                n_samples=n_samples, group_ndims=2)
                    w = tf.tile(w, [1, tf.shape(x)[0], 1, 1])
                    f = tf.concat([f, tf.ones([n_samples, tf.shape(x)[0], 1, 1])], 2)
                    f = tf.matmul(w, f) / tf.sqrt(self.layer_sizes[i]+1.)
                    if(i < len(self.layer_sizes)-2):
                        f = self.activation(f)
                y_mean = tf.squeeze(f, [2, 3])
                y_logstd = tf.get_variable('y_logstd', shape=[],
                                        initializer=tf.constant_initializer(0.))
                y = zs.Laplace('y', y_mean, scale=tf.exp(y_logstd))
            return model, y_mean
            
        @zs.reuse('variational')
        def variational(n_samples):
            with zs.BayesianNet() as variational:
                for i in range(len(self.layer_sizes)-1):
                    w_mean = tf.get_variable('w_mean_'+str(i),
                        shape=[1, self.layer_sizes[i+1], self.layer_sizes[i]+1],
                        initializer=tf.constant_initializer(0.))
                    w_logstd = tf.get_variable('w_logstd_'+str(i),
                    shape=[1, self.layer_sizes[i+1], self.layer_sizes[i]+1],
                        initializer=tf.constant_initializer(0.))
                    zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                            n_samples=n_samples, group_ndims=2)
            return variational
        
        