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
    w_names = []
    return w_names

@zs.reuse('model')
def p_Y_Xw(observed, X, drop_rate, n_basis, net_sizes, n_samples, task):
    with zs.BayesianNet(observed=observed) as model:
        f = tf.expand_dims(X, 1)
        for i in range(len(net_sizes)-1):
            f = tf.layers.dense(f, net_sizes[i+1],
                kernel_regularizer=layers.l2_regularizer(scale=1e-2),
                bias_regularizer=layers.l2_regularizer(scale=1e-2))
            if(i < len(net_sizes)-2):
                f = tf.nn.relu(f)
        f = tf.squeeze(f, [1])
        if(task == "classification"):
            f = tf.nn.softmax(f)
    return model, f, tf.losses.get_regularization_loss()