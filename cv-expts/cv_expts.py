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

import os
import time
import importlib
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.contrib.layers as layers
from six.moves import range, zip
import numpy as np
import zhusuan as zs


def standardize(data_train, data_test):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean)/std
    data_test_standardized = (data_test - mean)/std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std

def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :return: `uint8` numpy array
        The output image.
    """
    makedirs(filename)
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)

def gradient_ascent_attack(model_names, dataset_name, **args):
    np.random.seed(314159)
    
    # Define task
    task = 'classification'
    
    # Define model parameters
    n_basis = [50] if 'n_basis' not in args.keys() else args['n_basis']
    n_hiddens = [50] if 'n_hiddens' not in args.keys() else args['n_hiddens']

    # Define training/evaluation parameters
    IMG_W = 28 if 'img_w' not in args.keys() else args['img_w']
    IMG_H = 28 if 'img_h' not in args.keys() else args['img_h']
    N_CHANNELS = 28 if 'n_channels' not in args.keys() else args['n_channels']
    N_GEN_IMGS = 10 if 'n_gen_imgs' not in args.keys() else args['n_gen_imgs']
    GEN_CLASS = 0 if 'gen_class' not in args.keys() else args['gen_class']
    N_CLASS = 10 if 'n_class' not in args.keys() else args['n_class']
    DROP_RATE = 0.5 if 'drop_rate' not in args.keys() else args['drop_rate']
    N_SAMPLES = 500 if 'n_samples' not in args.keys() else args['n_samples']
    MAX_EPOCHS = 2000 if 'max_epochs' not in args.keys() else args['max_epochs']
    CHECK_FREQ = 5 if 'check_freq' not in args.keys() else args['check_freq']
    EARLY_STOP = 5 if 'early_stop' not in args.keys() else args['early_stop']
    LEARN_RATE = 1e-3 if 'learn_rate' not in args.keys() else args['learn_rate']
    
    D, N, P = IMG_W*IMG_H*N_CHANNELS, N_GEN_IMGS, N_CLASS
    net_sizes = [D]+n_hiddens+[P]

    # Build the computation graph
    n_samples = tf.placeholder(tf.int32, shape=[], name='n_samples')
    X = tf.get_variable('X', shape=[N_GEN_IMGS, IMG_W, IMG_H, N_CHANNELS],
        initializer=tf.random_normal_initializer())
    X_in = tf.reshape(X, [N_GEN_IMGS, IMG_W*IMG_H*N_CHANNELS])
    for model_name in model_names:

        module = importlib.import_module("models."+model_name)
        w_names = module.get_w_names(DROP_RATE, net_sizes)    
        model_code = model_name+"{"+",".join(list(map(str, net_sizes)))+"}"
        
        cost = 0
        observed = {}
        if(model_name != "DNN"):
            var = module.var_q_w(n_basis, net_sizes, n_samples)
            q_w_outputs = var.query(w_names,
                outputs=True, local_log_prob=True)
            latent = dict(zip(w_names, q_w_outputs))
            observed.update({
                (w_name, latent[w_name][0]) for w_name in w_names})
        
        # prediction: rms error & log likelihood
        model, f, reg_cost = module.p_Y_Xw(observed, X_in,
            DROP_RATE, n_basis, net_sizes, n_samples, task)
        if(model_name == "DNN"):
            y_pred = f
        else:
            y_pred, y_var = tf.nn.moments(f, axes=[0])
        cost = -tf.reduce_mean(y_pred[:, GEN_CLASS])
        if(model_name != "DNN"):
            cost += tf.reduce_mean(y_var[:, GEN_CLASS])
        
        learn_rate_ph = tf.placeholder(tf.float32, shape=[])
        global_step = tf.Variable(0, trainable=False)
        learn_rate_ts = tf.train.exponential_decay(
            learn_rate_ph, global_step, 10000, 0.96, staircase=True)
        if(model_name  == "VIBayesNN"):
            learn_rate_ts *= 10
        optimizer = tf.train.AdamOptimizer(learn_rate_ts)
        infer_op = optimizer.minimize(cost,
            var_list=[X], global_step=global_step)
    
        restore_vars = {}
        for var in tf.trainable_variables():
            if(var.name != 'X'):
                restore_vars[var.name] = var
    
        # Run the inference
        with tf.Session() as sess:
            X.initializer.run()
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, './trained/'+model_code+'_'+dataset_name)
            for epoch in range(MAX_EPOCHS):
                flag_cvrg = True
                time_epoch = -time.time()
                _, gen_cost, y_pred, y_var = sess.run(
                    [infer_op, cost], feed_dict={n_samples: N_SAMPLES,
                        learn_rate_ph: LEARN_RATE})
                time_epoch += time.time()
                print('Epoch {} ({:.1f}s, {}): Cost = {:.8f}'.format(
                    epoch, time_epoch, cnt_cvrg, gen_cost))
                if(best_cost > gen_cost):
                    flag_cvrg = False
                if epoch % CHECK_FREQ == 0 and epoch > 0:
                    X_gen, gen_cost, p_pred, p_var = sess.run(
                        [X, cost, y_pred, y_var],
                            feed_dict={n_samples: N_SAMPLES})
                    if not os.path.exists('./gen_images/'):
                        os.makedirs('./gen_images/')
                    print('Epoch {} : Softmax output = {} \pm {}'.
                        format(epoch, p_pred, p_var))
                    gen_path = './gen_images/{}.epoch.{}.iter.{}.png'.format(
                        model_name, epoch, iter)
                    save_image_collections(X_gen, gen_path, scale_each=True)
                    if(not flag_cvrg):
                        cnt_cvrg = 0
                    if(best_cost < gen_cost):
                        cnt_cvrg += 1
                    else:
                        best_epoch = len(f_gen_costs)
                        cnt_cvrg = 0
                        best_cost = gen_cost
                    if(cnt_cvrg > EARLY_STOP-(epoch*EARLY_STOP/MAX_EPOCHS)):
                        break
                
    return eval_tms, eval_lls
