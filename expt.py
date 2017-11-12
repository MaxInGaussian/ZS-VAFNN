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
import pandas as pd
import numpy as np
import zhusuan as zs


def standardize(data_train, data_valid, data_test):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    train_standardized = (data_train - mean)/std
    valid_standardized = (data_test - mean)/std
    test_standardized = (data_test - mean)/std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return train_standardized, valid_standardized, test_standardized, mean, std

def run_experiment(model_names, dataset_name, dataset, **args):
    
    # Define task
    task = 'regression' if 'task' not in args.keys() else args['task']
    
    # Define model parameters
    n_basis = [50] if 'n_basis' not in args.keys() else args['n_basis']
    n_hiddens = [50] if 'n_hiddens' not in args.keys() else args['n_hiddens']

    # Define training/evaluation parameters
    SAVE = False if 'save' not in args.keys() else args['save']
    PLOT = True if 'plot' not in args.keys() else args['plot']
    DROP_RATE = 0.5 if 'drop_rate' not in args.keys() else args['drop_rate']
    TRAIN_SAMPLES = 20 if 'train_samples' not in args.keys() else args['train_samples']
    TEST_SAMPLES = 100 if 'test_samples' not in args.keys() else args['test_samples']
    MAX_ITERS = 500 if 'max_iters' not in args.keys() else args['max_iters']
    BATCH_SIZE = 50 if 'batch_size' not in args.keys() else args['batch_size']
    MAX_EPOCHS = 2000 if 'max_epochs' not in args.keys() else args['max_epochs']
    CHECK_FREQ = 5 if 'check_freq' not in args.keys() else args['check_freq']
    EARLY_STOP = 5 if 'early_stop' not in args.keys() else args['early_stop']
    LEARN_RATE = 1e-3 if 'learn_rate' not in args.keys() else args['learn_rate']

    N, D = dataset[0][0].shape
    T, P = dataset[0][-1].shape
    net_sizes = [D]+n_hiddens+[P]
    
    min_tm, max_nll = np.Infinity, -np.Infinity
    eval_tms = {model_name:[] for model_name in model_names}
    eval_lls = {model_name:[] for model_name in model_names}
    
    np.random.seed(314159)
    tf.set_random_seed(314159)

    # Build the computation graph
    n_samples = tf.placeholder(tf.int32, shape=[], name='n_samples')
    X = tf.placeholder(tf.float32, shape=[None, D])
    if(task == "regression"):
        y = tf.placeholder(tf.float32, shape=[None, P])
    elif(task == "classification"):
        y = tf.placeholder(tf.int32, shape=[None, P])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_samples, 1, 1])
        
    for fold in range(len(dataset)):
            
        X_train, y_train, X_valid, y_valid, X_test, y_test = dataset[fold]
        problem_name = dataset_name.replace(' ', '_')+'_'+str(fold+1)
        N, M, T = X_train.shape[0], X_valid.shape[0], X_test.shape[0]
        train_iters = int(np.floor(N/float(BATCH_SIZE)))
        valid_iters = int(np.floor(M/float(BATCH_SIZE)))
        test_iters = int(np.floor(T/float(BATCH_SIZE)))
    
        # Standardize data
        X_train, X_valid, X_test = standardize(X_train, X_valid, X_test)[:3]
        if(task == "regression"):
            y_train, y_valid, y_test, mean_y_train, std_y_train =\
                standardize(y_train, y_valid, y_test)
        
        for model_name in model_names:
    
            module = importlib.import_module("models."+model_name)
            w_names = module.get_w_names(DROP_RATE, net_sizes)     
            model_code = model_name+"{"+",".join(list(map(str, net_sizes)))+"}"
            
            observed = {'y': y_obs}
            if(len(w_names) > 0):
                var = module.var_q_w(n_basis, net_sizes, n_samples)
                q_w_outputs = var.query(w_names,
                    outputs=True, local_log_prob=True)
                latent = dict(zip(w_names, q_w_outputs))
                observed.update({
                    (w_name, latent[w_name][0]) for w_name in w_names})
    
                if('VI' in model_name):
                    def log_joint(observed):
                        model, _, _ = module.p_Y_Xw(observed, X, DROP_RATE,
                            n_basis, net_sizes, n_samples, task)
                        log_py_xw = model.local_log_prob('y')
                        log_j = zs.log_mean_exp(log_py_xw, 0)*N
                        if(len(w_names)):
                            log_pws = model.local_log_prob(w_names)
                            log_j += tf.add_n(log_pws)
                        return log_j
                    lower_bound = zs.variational.elbo(
                        log_joint, observed={'y': y_obs}, latent=latent, axis=0)
                    cost = tf.reduce_mean(lower_bound.sgvb())
                    lower_bound = tf.reduce_mean(lower_bound)       
            
            # Prediction: rms error & nlpd
            model, f, side_prod = module.p_Y_Xw(observed, X,
                DROP_RATE, n_basis, net_sizes, n_samples, task)
            if(model_name == "DNN"):
                y_pred, y_var = f, 1.
            else:
                y_pred, y_var = tf.nn.moments(f, axes=[0])
                if(task == "classification"):
                    g_mu, g_var = tf.nn.moments(f, axes=[2])
                    g_mu = tf.expand_dims(tf.reduce_mean(g_mu, 0), 1)
                    g_var = tf.expand_dims(tf.reduce_mean(g_var, 0), 1)
                    y_pred = (y_pred-g_mu)/g_var**0.5
                    y_pred = tf.contrib.distributions.Normal(0., 1.).cdf(y_pred)
            if(model_name == "DNN"):
                if(task == "regression"):
                    cost = tf.losses.mean_squared_error(y_pred, y)
                if(task == "classification"):
                    cost = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=y_pred, labels=y))
            elif('MC' in model_name):
                cost = tf.losses.mean_squared_error(y_pred, y)
            else:
                log_py_xw = model.local_log_prob('y')
            if(side_prod is not None):
                cost += side_prod
            
            learn_rate_ph = tf.placeholder(tf.float32, shape=[])
            global_step = tf.Variable(0, trainable=False)
            learn_rate_ts = tf.train.exponential_decay(
                learn_rate_ph, global_step, 10000, 0.96, staircase=True)
            if(model_name  == "VIBayesNN"):
                learn_rate_ts *= 5
            optimizer = tf.train.AdamOptimizer(learn_rate_ts)
            infer_op = optimizer.minimize(cost, global_step=global_step)
            
            if(SAVE):
                save_vars = {}
                for var in tf.trainable_variables():
                    print(var.name, var.get_shape())
                    save_vars[var.name] = var
                saver = tf.train.Saver(save_vars)
                save_path = './trained/'+model_code+'_'+dataset_name+'.ckpt'
            else:
                save_vars = []
                for var in tf.trainable_variables():
                    save_vars.append(tf.Variable(var.initialized_value()))
                assign_to_save, assign_to_restore = [], []
                for var, save_var in zip(tf.trainable_variables(), save_vars):
                    assign_to_save.append(save_var.assign(var))
                    assign_to_restore.append(var.assign(save_var))
            
            # Define optimization objective
            if(task == "regression"):
                NLPD = 0.5*tf.reduce_mean(tf.log(y_var*std_y_train**2)+((
                    y-y_pred)**2)/y_var)+0.5*np.log(2*np.pi)
                LL = NLPD
                rms_error = tf.sqrt(tf.reduce_mean((y_pred - y)**2))*std_y_train
                task_measure = tf.reduce_mean(rms_error)
            elif(task == "classification"):
                AUC = 0.
                for p in range(P):
                    AUC += tf.metrics.auc(
                        labels=y[:, p], predictions=y_pred[:, p])[1]
                LL = AUC/P
                y_pred = tf.argmax(y_pred, 1)
                sparse_y = tf.argmax(y, 1)
                accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(y_pred, sparse_y), tf.float32))
                task_measure = accuracy
                    
            # Run the inference
            def get_batch(X_data, y_data, t, iters):
                if(iters <= MAX_ITERS):
                    if(t == iters-1):                        
                        X_batch = X_data[t*BATCH_SIZE:]
                        y_batch = y_data[t*BATCH_SIZE:]
                    else:
                        X_batch = X_data[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                        y_batch = y_data[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                else:
                    inds = np.random.choice(range(N), BATCH_SIZE, replace=False)
                    X_batch, y_batch = X_data[inds], y_data[inds]
                return X_batch, y_batch
            best_cost, cnt_cvrg = np.Infinity, 0
            if(task == 'regression'):
                best_tm, best_ll = np.Infinity, np.Infinity
            elif(task == 'classification'):
                best_tm, best_ll = -np.Infinity, -np.Infinity
            valid_costs, valid_tms, valid_lls = [], [], []
            test_costs, test_tms, test_lls = [], [], []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                for epoch in range(MAX_EPOCHS):
                    time_epoch = -time.time()
                    costs = []
                    for iter in range(min(MAX_ITERS, train_iters)):
                        Xt_batch, yt_batch = get_batch(
                            X_train, y_train, iter, train_iters)
                        _, c = sess.run(
                            [infer_op, cost],
                            feed_dict={n_samples: TRAIN_SAMPLES,
                                learn_rate_ph: LEARN_RATE,
                                X: Xt_batch, y: yt_batch})
                        costs.append(c)
                    time_epoch += time.time()
                    train_cost =  np.mean(costs)
                    print('Epoch {} ({:.1f}s, {}): Cost = {:.8f}'.format(
                        epoch, time_epoch, cnt_cvrg, train_cost))
                    if epoch % CHECK_FREQ == 0 and epoch > 0:
                        costs, tms, lls = [], [], []
                        time_valid = -time.time()
                        for iter in range(valid_iters):
                            Xv_batch, yv_batch = get_batch(
                                X_valid, y_valid, iter, valid_iters)
                            c, tm, ll = sess.run([cost, task_measure, LL],
                                feed_dict={n_samples: TEST_SAMPLES,
                                    X: Xv_batch, y: yv_batch})
                            costs.append(c);tms.append(tm);lls.append(ll)
                        time_valid += time.time()
                        valid_cost, valid_tm, valid_ll =\
                            np.mean(costs), np.mean(tms), np.mean(lls)
                        if(len(valid_costs)==0):
                            valid_costs.append(valid_cost)
                            valid_tms.append(valid_tm)
                            valid_lls.append(valid_ll)
                        else:
                            valid_costs.append(valid_cost)
                            valid_tms.append(valid_tm)
                            valid_lls.append(valid_ll)
                        print('>>>', model_code, '>>>', problem_name)
                        print('>>>> Validation ({:.1f}s) - best = {:.8f}'.format(
                            time_valid, best_cost))
                        print('>> Valid Cost = {:.8f}'.format(valid_cost))
                        if(task == "regression"):
                            print('>> Valid RMSE = {:.8f}'.format(valid_tm))
                            print('>> Valid NLPD = {:.8f}'.format(valid_ll))
                        elif(task == "classification"):
                            print('>> Valid ACC = {:.8f}'.format(valid_tm))
                            print('>> Valid AUC = {:.8f}'.format(valid_ll))
                        tms, lls = [], []
                        time_test = -time.time()
                        for t in range(test_iters):
                            if(t == test_iters-1):
                                X_batch = X_test[t*BATCH_SIZE:]
                                y_batch = y_test[t*BATCH_SIZE:]
                            else:
                                X_batch = X_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                                y_batch = y_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                            tm, ll = sess.run([task_measure, LL],
                                feed_dict={n_samples: TEST_SAMPLES,
                                    X: X_batch, y: y_batch})
                            tms.append(tm);lls.append(ll)
                        test_tm, test_ll = np.mean(tms), np.mean(lls)
                        time_test += time.time()
                        if(len(test_tms)==0):
                            test_tms.append(test_tm)
                            test_lls.append(test_ll)
                        else:
                            test_tms.append(test_tm)
                            test_lls.append(test_ll)
                        print('>>>> TEST ({:.1f}s)'.format(time_test))
                        if(task == "regression"):
                            print('>> Test RMSE = {:.8f}'.format(test_tm))
                            print('>> Test NLPD = {:.8f}'.format(test_ll))
                        elif(task == "classification"):
                            print('>> Test ACC = {:.8f}'.format(test_tm))
                            print('>> Test AUC = {:.8f}'.format(test_ll))
                        if(((best_tm > valid_tm or best_ll > valid_ll)
                            and task=='regression') or
                            ((best_tm < valid_tm or best_ll < valid_ll)
                            and task=='classification')):
                            print('!!!! NEW BEST IN VALID !!!!')
                            cnt_cvrg = 0
                            best_tm = valid_tm
                            best_ll = valid_ll
                            best_cost = valid_cost
                            if(SAVE):
                                if not os.path.exists('./trained/'):
                                    os.makedirs('./trained/')
                                saver.save(sess, save_path)
                            else:
                                sess.run(assign_to_save)
                        else:
                            cnt_cvrg += 1
                        if(cnt_cvrg > EARLY_STOP-(epoch*EARLY_STOP/MAX_EPOCHS)):
                            break
                
                # Load the selected best params and evaluate its performance
                if(SAVE):
                    saver.restore(sess, save_path)
                else:
                    sess.run(assign_to_restore)
                tms, lls = [], []
                for t in range(test_iters):
                    if(t == test_iters-1):
                        X_batch = X_test[t*BATCH_SIZE:]
                        y_batch = y_test[t*BATCH_SIZE:]
                    else:
                        X_batch = X_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                        y_batch = y_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                    tm, ll = sess.run([task_measure, LL],
                        feed_dict={n_samples: TEST_SAMPLES,
                            X: X_batch, y: y_batch})
                    tms.append(tm);lls.append(ll)
                test_tm, test_ll = np.mean(tms), np.mean(lls)
                print('>>> BEST TEST')
                if(task == "regression"):
                    print('>> Test RMSE = {:.8f}'.format(test_tm))
                    print('>> Test NLPD = {:.8f}'.format(test_ll))
                elif(task == "classification"):
                    print('>> Test ACC = {:.8f}'.format(test_tm))
                    print('>> Test AUC = {:.8f}'.format(test_ll))
                eval_tms[model_name].append(test_tm)
                eval_lls[model_name].append(test_ll)
            if(PLOT):
                import matplotlib.pyplot as plt
                plt.figure()
                plt.subplot(3, 1, 1)        
                plt.title(model_code+" on "+dataset_name)
                test_MAX_EPOCHS = np.arange(len(valid_costs))*CHECK_FREQ
                plt.semilogx(test_MAX_EPOCHS, valid_costs, '--', label='Train')
                plt.xlabel('Epoch')
                plt.ylabel('Cost')
                plt.subplot(3, 1, 2)
                plt.semilogx(test_MAX_EPOCHS, valid_tms, '--', label='Train')
                plt.semilogx(test_MAX_EPOCHS, test_tms, label='Test')
                plt.legend(loc='lower left')
                plt.xlabel('Epoch')
                if(task == "regression"):
                    plt.ylabel('RMSE {:.4f}'.format(test_tm))
                elif(task == "classification"):
                    plt.ylabel('ACC {:.4f}'.format(test_tm))
                plt.subplot(3, 1, 3)
                plt.semilogx(test_MAX_EPOCHS, valid_lls, '--', label='Train')
                plt.semilogx(test_MAX_EPOCHS, test_lls, label='Test')
                plt.xlabel('Epoch')
                if(task == "regression"):
                    plt.ylabel('NLPD {:.4f}'.format(test_ll))
                elif(task == "classification"):
                    plt.ylabel('AUC {:.4f}'.format(test_ll))
                if not os.path.exists('./plots/'):
                    os.makedirs('./plots/')
                plt.savefig('./plots/'+model_code+'_'+problem_name+'.png')
                plt.close()
        tm_mu = np.mean(eval_tms[model_name])
        tm_std = np.std(eval_tms[model_name])
        ll_mu = np.mean(eval_lls[model_name])
        ll_std = np.std(eval_lls[model_name])
        tm_name = 'RMSE' if task == "regression" else 'ACC'
        ll_name = 'NLPD' if task == "regression" else 'AUC'
        res_cols = [tm_name+' (mean)', tm_name+' (1.96*std)', tm_name,
            ll_name+' (mean)', ll_name+' (1.96*std)', ll_name]
        if not os.path.isfile('result.csv'):
            df_res = pd.DataFrame(columns=res_cols)
        else:
            df_res = pd.DataFrame.from_csv(
                path='result.csv', header=0, index_col=0)
        df_res.loc[model_code] = [tm_mu, 1.96*tm_std,
            '{:.4f}\pm{:.4f}'.format(tm_mu, 1.96*tm_std), ll_mu,
            1.96*ll_std, '{:.4f}\pm{:.4f}'.format(ll_mu, 1.96*ll_std)]
        df_res.to_csv('result.csv', header=res_cols)
    return eval_tms, eval_lls