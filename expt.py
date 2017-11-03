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

def run_experiment(model_names, dataset_name, train_test_set, **args):
    
    # Define task
    task = 'regression' if 'task' not in args.keys() else args['task']
    
    # Define model parameters
    n_basis = [50] if 'n_basis' not in args.keys() else args['n_basis']
    n_hiddens = [50] if 'n_hiddens' not in args.keys() else args['n_hiddens']

    # Define training/evaluation parameters
    save = False if 'save' not in args.keys() else args['save']
    plot = True if 'plot' not in args.keys() else args['plot']
    drop_rate = 0.5 if 'drop_rate' not in args.keys() else args['drop_rate']
    lb_samples = 20 if 'lb_samples' not in args.keys() else args['lb_samples']
    ll_samples = 100 if 'll_samples' not in args.keys() else args['ll_samples']
    batch_size = 50 if 'batch_size' not in args.keys() else args['batch_size']
    max_epochs = 2000 if 'max_epochs' not in args.keys() else args['max_epochs']
    check_freq = 5 if 'check_freq' not in args.keys() else args['check_freq']
    early_stop = 5 if 'early_stop' not in args.keys() else args['early_stop']
    lr = 1e-3 if 'lr' not in args.keys() else args['lr']
    
    D, P = train_test_set[0][0].shape[1], train_test_set[0][1].shape[1]
    net_sizes = [D]+n_hiddens+[P]
    
    min_tm, max_nll = np.Infinity, -np.Infinity
    eval_tms = {model_name:[] for model_name in model_names}
    eval_lls = {model_name:[] for model_name in model_names}
    train_lbs, train_tms, train_lls = [], [], []
    test_lbs, test_tms, test_lls = [], [], []
    for fold, (X_train, y_train, X_test, y_test) in enumerate(train_test_set):
    
        problem_name = dataset_name.replace(' ', '_')+'_'+str(fold+1)
        N, T = X_train.shape[0], X_test.shape[0]
        iters = int(np.floor(N/float(batch_size)))
        t_iters = int(np.floor(T/float(batch_size)))
    
        # Standardize data
        X_train, X_test, _, _ = standardize(X_train, X_test)
        if(task == "regression"):
            y_train, y_test, mean_y_train, std_y_train =\
                standardize(y_train, y_test)
    
    
        # Build the computation graph
        n_samples = tf.placeholder(tf.int32, shape=[], name='n_samples')
        X = tf.placeholder(tf.float32, shape=[None, D])
        if(task == "regression"):
            y = tf.placeholder(tf.float32, shape=[None, P])
        elif(task == "classification"):
            y = tf.placeholder(tf.int32, shape=[None, P])
        y_obs = tf.tile(tf.expand_dims(y, 0), [n_samples, 1, 1])
    
        for model_name in model_names:
    
            module = importlib.import_module("models."+model_name)
            w_names = module.get_w_names(drop_rate, net_sizes)    
            model_code = model_name+"{"+",".join(list(map(str, net_sizes)))+"}"
            
            def log_joint(observed):
                model, _, _ = module.p_Y_Xw(observed, X, drop_rate, n_basis,
                    net_sizes, n_samples, task)
                log_py_xw = model.local_log_prob('y')
                log_j = zs.log_mean_exp(log_py_xw, 0)*N
                if(len(w_names)):
                    log_pws = model.local_log_prob(w_names)
                    log_j += tf.add_n(log_pws)
                return log_j

            observed = {'y': y_obs}
            if(len(w_names)):
                var = module.var_q_w(n_basis, net_sizes, n_samples)
                q_w_outputs = var.query(w_names,
                    outputs=True, local_log_prob=True)
                latent = dict(zip(w_names, q_w_outputs))
            
                lower_bound = zs.variational.elbo(
                    log_joint, observed={'y': y_obs}, latent=latent, axis=0)
                cost = tf.reduce_mean(lower_bound.sgvb())
                lower_bound = tf.reduce_mean(lower_bound)
                observed.update({
                    (w_name, latent[w_name][0]) for w_name in w_names})
            
            # prediction: rms error & log likelihood
            model, ys, reg_cost = module.p_Y_Xw(observed, X,
                drop_rate, n_basis, net_sizes, n_samples, task)
            if(reg_cost is not None):
                cost += reg_cost
            y_pred = tf.reduce_mean(ys, 0)
            if(task == "regression"):
                rms_error = tf.sqrt(tf.reduce_mean((y_pred - y)**2))*std_y_train
                task_measure = rms_error
            elif(task == "classification"):
                y_pred = tf.argmax(y_pred, 1)
                sparse_y = tf.argmax(y, 1)
                accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(y_pred, sparse_y), tf.float32))
                task_measure = 1-accuracy
            log_py_xw = model.local_log_prob('y')
            if(task == "regression"):
                log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))-\
                    tf.log(std_y_train)
            elif(task == "classification"):
                log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))
            
            if(model_name == "DNN"):
                cost = task_measure
                lower_bound = -task_measure
            
            lr_ph = tf.placeholder(tf.float32, shape=[])
            global_step = tf.Variable(0, trainable=False)
            lr_ts = tf.train.exponential_decay(
                lr_ph, global_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(lr_ts)
            infer_op = optimizer.minimize(cost, global_step=global_step)
        
            params = tf.trainable_variables()
            for i in params:
                print(i.name, i.get_shape())
        
            # Run the inference
            best_epoch, best_lb, count_over_train = 0, -np.Infinity, 0
            f_train_lbs, f_train_tms, f_train_lls = [], [], []
            f_test_lbs, f_test_tms, f_test_lls = [], [], []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(max_epochs):
                    time_epoch = -time.time()
                    lbs = []
                    for t in range(iters):
                        if(t == iters-1):                        
                            X_batch = X_train[t*batch_size:]
                            y_batch = y_train[t*batch_size:]
                        else:
                            X_batch = X_train[t*batch_size:(t+1)*batch_size]
                            y_batch = y_train[t*batch_size:(t+1)*batch_size]
                        _, lb = sess.run(
                            [infer_op, cost],
                            feed_dict={n_samples: lb_samples,
                                lr_ph: lr, X: X_batch, y: y_batch})
                        lbs.append(lb)
                    time_epoch += time.time()
                    print('Epoch {} ({:.1f}s, {}): Cost = {:.8f}'.format(
                        epoch, time_epoch, count_over_train, np.mean(lbs)))
                    if epoch % check_freq == 0:
                        lbs, tms, lls = [], [], []
                        time_train = -time.time()
                        for t in range(iters):
                            if(t == iters-1):                        
                                X_batch = X_train[t*batch_size:]
                                y_batch = y_train[t*batch_size:]
                            else:
                                X_batch = X_train[t*batch_size:(t+1)*batch_size]
                                y_batch = y_train[t*batch_size:(t+1)*batch_size]
                            lb, tm, ll = sess.run(
                                [lower_bound, task_measure, log_likelihood],
                                feed_dict={n_samples: ll_samples,
                                    X: X_batch, y: y_batch})
                            lbs.append(lb);tms.append(tm);lls.append(ll)
                        train_lb, train_tm, train_ll =\
                            np.mean(lbs), np.mean(tms), np.mean(lls)
                        time_train += time.time()
                        if(len(f_train_lbs)==0):
                            f_train_lbs.append(train_lb)
                            f_train_tms.append(train_tm)
                            f_train_lls.append(train_ll)
                        else:
                            f_train_lbs.append(train_lb)
                            f_train_tms.append(train_tm)
                            f_train_lls.append(train_ll)
                        print('>>> TRAIN ({:.1f}s) - best_lb = {:.8f}'.format(
                            time_train, best_lb))
                        print('>> Train lower bound = {:.8f}'.format(train_lb))
                        print('>> Train log_likelihood = {:.8f}'.format(train_ll))
                        if(task == "regression"):
                            print('>> Train rmse = {:.8f}'.format(train_tm))
                        elif(task == "classification"):
                            print('>> Train err_rate = {:.8f}'.format(train_tm))
                        lbs, tms, lls = [], [], []
                        time_test = -time.time()
                        for t in range(t_iters):
                            if(t == t_iters-1):                   
                                X_batch = X_test[t*batch_size:]
                                y_batch = y_test[t*batch_size:]
                            else:
                                X_batch = X_test[t*batch_size:(t+1)*batch_size]
                                y_batch = y_test[t*batch_size:(t+1)*batch_size]
                            lb, mse, ll = sess.run(
                                [lower_bound, task_measure, log_likelihood],
                                feed_dict={n_samples: ll_samples,
                                X: X_batch, y: y_batch})
                            lbs.append(lb);tms.append(mse);lls.append(ll)
                        test_lb, test_tm, test_ll =\
                            np.mean(lbs), np.mean(tms), np.mean(lls)
                        time_test += time.time()
                        if(len(f_test_lbs)==0):
                            f_test_lbs.append(test_lb)
                            f_test_tms.append(test_tm)
                            f_test_lls.append(test_ll)
                        else:
                            f_test_lbs.append(test_lb)
                            f_test_tms.append(test_tm)
                            f_test_lls.append(test_ll)
                        print('>>> TEST ({:.1f}s)'.format(time_test))
                        print('>> Test lower bound = {:.8f}'.format(test_lb))
                        print('>> Test log_likelihood = {:.8f}'.format(test_ll))
                        if(task == "regression"):
                            print('>> Test rmse = {:.8f}'.format(test_tm))
                        elif(task == "classification"):
                            print('>> Test err_rate = {:.8f}'.format(test_tm))
                        if(best_lb < train_lb):
                            best_epoch = len(f_train_lbs)
                            count_over_train = 0
                            best_lb = train_lb
                            if(save):
                                saver = tf.train.Saver()
                                if not os.path.exists('./trained/'):
                                    os.makedirs('./trained/')
                                saver.save(sess,
                                    './trained/'+model_code+'_'+problem_name)
                            else:
                                min_tm, max_ll = test_tm, test_ll
                        else:
                            count_over_train += 1
                        if(count_over_train > early_stop):
                            break
                
                # Load the selected best params and evaluate its performance
                if(save):
                    saver = tf.train.Saver()
                    saver.restore(sess, './trained/'+model_code+'_'+problem_name)
                    tms, lls = [], []
                    t_iters = int(np.floor(X_test.shape[0]/float(batch_size)))
                    for t in range(t_iters):
                        if(t == t_iters-1):                   
                            X_batch = X_test[t*batch_size:]
                            y_batch = y_test[t*batch_size:]
                        else:
                            X_batch = X_test[t*batch_size:(t+1)*batch_size]
                            y_batch = y_test[t*batch_size:(t+1)*batch_size]
                        mse, ll = sess.run(
                            [rms_error, log_likelihood],
                            feed_dict={n_samples: ll_samples, drop_rate: False,
                                X: X_batch, y: y_batch})
                        tms.append(mse);lls.append(ll)
                    test_tm, test_ll = np.mean(tms), np.mean(lls)
                else:
                    test_tm, test_ll = min_tm, max_ll
                print('>>> BEST TEST')
                print('>> Test log_likelihood = {:.8f}'.format(test_ll))
                if(task == "regression"):
                    print('>> Test rmse = {:.8f}'.format(test_tm))
                elif(task == "classification"):
                    print('>> Test err_rate = {:.8f}'.format(test_tm))
                eval_tms[model_name].append(test_tm)
                eval_lls[model_name].append(test_ll)
            if(plot):
                import matplotlib.pyplot as plt
                plt.figure()
                plt.subplot(3, 1, 1)        
                plt.title(model_code+" on "+dataset_name)
                test_max_epochs = np.arange(len(f_train_lbs))*check_freq
                plt.semilogx(test_max_epochs, f_train_lbs, '--', label='Train')
                plt.semilogx(test_max_epochs, f_test_lbs, label='Test')
                plt.xlabel('Epoch')
                plt.ylabel('ELBO {:.4f}'.format(test_lb))
                plt.subplot(3, 1, 2)
                plt.semilogx(test_max_epochs, f_train_tms, '--', label='Train')
                plt.semilogx(test_max_epochs, f_test_tms, label='Test')
                plt.legend(loc='lower left')
                plt.xlabel('Epoch')
                if(task == "regression"):
                    plt.ylabel('RMSE {:.4f}'.format(test_tm))
                elif(task == "classification"):
                    plt.ylabel('CERR {:.4f}'.format(test_tm))
                    plt.ylim([0, 1])
                plt.subplot(3, 1, 3)
                plt.semilogx(test_max_epochs, f_train_lls, '--', label='Train')
                plt.semilogx(test_max_epochs, f_test_lls, label='Test')
                plt.xlabel('Epoch')
                plt.ylabel('Log Likelihood {:.4f}'.format(test_ll))
                if not os.path.exists('./plots/'):
                    os.makedirs('./plots/')
                plt.savefig('./plots/'+model_code+'_'+problem_name+'.png')
                plt.close()
            train_lbs.append(np.array(f_train_lbs))
            train_tms.append(np.array(f_train_tms))
            train_lls.append(np.array(f_train_lls))
            test_lbs.append(np.array(f_test_lbs))
            test_tms.append(np.array(f_test_tms))
            test_lls.append(np.array(f_test_lls))
    return eval_tms, eval_lls