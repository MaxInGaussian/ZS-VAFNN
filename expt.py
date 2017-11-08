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
    np.random.seed(314159)
    
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
    
    D, P = train_test_set[0][0].shape[1], train_test_set[0][1].shape[1]
    net_sizes = [D]+n_hiddens+[P]
    
    min_tm, max_nll = np.Infinity, -np.Infinity
    eval_tms = {model_name:[] for model_name in model_names}
    eval_lls = {model_name:[] for model_name in model_names}
    train_costs, train_tms, train_lls = [], [], []
    test_costs, test_tms, test_lls = [], [], []
    for fold, (X_train, y_train, X_test, y_test) in enumerate(train_test_set):
    
        problem_name = dataset_name.replace(' ', '_')+'_'+str(fold+1)
        N, T = X_train.shape[0], X_test.shape[0]
        iters = int(np.floor(N/float(BATCH_SIZE)))
        t_iters = int(np.floor(T/float(BATCH_SIZE)))
    
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
            w_names = module.get_w_names(DROP_RATE, net_sizes)    
            model_code = model_name+"{"+",".join(list(map(str, net_sizes)))+"}"
            
            cost = 0
            observed = {'y': y_obs}
            if(model_name != "DNN"):
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
            
            # prediction: rms error & log likelihood
            model, f, reg_cost = module.p_Y_Xw(observed, X,
                DROP_RATE, n_basis, net_sizes, n_samples, task)
            if(model_name == "DNN"):
                y_pred = f
            else:
                y_pred, y_var = tf.nn.moments(f, axes=[0])
            if(model_name == "DNN" or 'MC' in model_name):
                if(task == "regression"):
                    cost = tf.losses.mean_squared_error(y_pred, y)
                elif(task == "classification"):
                    cost = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=y_pred, labels=y))
                lower_bound = -cost
            else:
                log_py_xw = model.local_log_prob('y')
            if(task == "regression"):
                if(model_name  == "DNN"):
                    LL = -cost
                else:
                    NLPD = 0.5*tf.reduce_mean(tf.log(y_var*std_y_train**2)+((
                        y-y_pred)**2)/y_var)+0.5*np.log(2*np.pi)
                    LL = NLPD
                rms_error = tf.sqrt(tf.reduce_mean((y_pred - y)**2))*std_y_train
                task_measure = rms_error
            elif(task == "classification"):
                AUC = tf.metrics.auc(labels=y, predictions=y_pred)
                if(model_name == "DNN"):
                    y_pred = tf.argmax(y_pred, 1)
                else:
                    y_pred = tf.argmax(tf.reduce_sum(
                        tf.one_hot(tf.argmax(f, 2), P), 0), 1)
                sparse_y = tf.argmax(y, 1)
                LL = AUC
                accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(y_pred, sparse_y), tf.float32))
                task_measure = 1-accuracy
            if(reg_cost is not None):
                cost += reg_cost
            
            learn_rate_ph = tf.placeholder(tf.float32, shape=[])
            global_step = tf.Variable(0, trainable=False)
            learn_rate_ts = tf.train.exponential_decay(
                learn_rate_ph, global_step, 10000, 0.96, staircase=True)
            if(model_name  == "VIBayesNN"):
                learn_rate_ts *= 10
            optimizer = tf.train.AdamOptimizer(learn_rate_ts)
            infer_op = optimizer.minimize(cost, global_step=global_step)
        
            params = tf.trainable_variables()
            for i in params:
                print(i.name, i.get_shape())
        
            # Run the inference
            def get_batch(t, iters):
                if(iters <= MAX_ITERS):
                    if(t == iters-1):                        
                        X_batch = X_train[t*BATCH_SIZE:]
                        y_batch = y_train[t*BATCH_SIZE:]
                    else:
                        X_batch = X_train[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                        y_batch = y_train[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                else:
                    inds = np.random.choice(range(N), BATCH_SIZE, replace=False)
                    X_batch, y_batch = X_train[inds], y_train[inds]
                return X_batch, y_batch
            best_epoch, best_cost, cnt_cvrg, flag_cvrg = 0, np.Infinity, 0, True
            f_train_costs, f_train_tms, f_train_lls = [], [], []
            f_test_costs, f_test_tms, f_test_lls = [], [], []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                for epoch in range(MAX_EPOCHS):
                    flag_cvrg = True
                    time_epoch = -time.time()
                    costs = []
                    for t in range(min(MAX_ITERS, iters)):
                        X_batch, y_batch = get_batch(t, iters)
                        _, c = sess.run(
                            [infer_op, cost],
                            feed_dict={n_samples: TRAIN_SAMPLES,
                                learn_rate_ph: LEARN_RATE,
                                X: X_batch, y: y_batch})
                        costs.append(c)
                    time_epoch += time.time()
                    train_cost =  np.mean(costs)
                    print('Epoch {} ({:.1f}s, {}): Cost = {:.8f}'.format(
                        epoch, time_epoch, cnt_cvrg, train_cost))
                    if(best_cost > train_cost):
                        flag_cvrg = False
                    if epoch % CHECK_FREQ == 0 and epoch > 0:
                        costs, tms, lls = [], [], []
                        time_train = -time.time()
                        for t in range(iters):
                            if(t == iters-1):                        
                                X_batch = X_train[t*BATCH_SIZE:]
                                y_batch = y_train[t*BATCH_SIZE:]
                            else:
                                X_batch = X_train[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                                y_batch = y_train[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                            c, tm, ll = sess.run([cost, task_measure, LL],
                                feed_dict={n_samples: TRAIN_SAMPLES,
                                    X: X_batch, y: y_batch})
                            costs.append(c);tms.append(tm);lls.append(ll)
                        train_cost, train_tm, train_ll =\
                            np.mean(costs), np.mean(tms), np.mean(lls)
                        time_train += time.time()
                        if(len(f_train_costs)==0):
                            f_train_costs.append(train_cost)
                            f_train_tms.append(train_tm)
                            f_train_lls.append(train_ll)
                        else:
                            f_train_costs.append(train_cost)
                            f_train_tms.append(train_tm)
                            f_train_lls.append(train_ll)
                        print('>>>', model_code, '>>>', problem_name)
                        print('>>> TRAIN ({:.1f}s) - best_cost = {:.8f}'.format(
                            time_train, best_cost))
                        print('>> Train lower bound = {:.8f}'.format(train_cost))
                        if(task == "regression"):
                            print('>> Train RMSE = {:.8f}'.format(train_tm))
                            print('>> Train NLPD = {:.8f}'.format(train_ll))
                        elif(task == "classification"):
                            print('>> Train Err Rate = {:.8f}'.format(train_tm))
                            print('>> Train AUC = {:.8f}'.format(train_ll))
                        costs, tms, lls = [], [], []
                        time_test = -time.time()
                        for t in range(t_iters):
                            if(t == t_iters-1):                   
                                X_batch = X_test[t*BATCH_SIZE:]
                                y_batch = y_test[t*BATCH_SIZE:]
                            else:
                                X_batch = X_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                                y_batch = y_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                            c, mse, ll = sess.run(
                                [cost, task_measure, LL],
                                feed_dict={n_samples: TEST_SAMPLES,
                                X: X_batch, y: y_batch})
                            costs.append(c);tms.append(mse);lls.append(ll)
                        test_cost, test_tm, test_ll =\
                            np.mean(costs), np.mean(tms), np.mean(lls)
                        time_test += time.time()
                        if(len(f_test_costs)==0):
                            f_test_costs.append(test_cost)
                            f_test_tms.append(test_tm)
                            f_test_lls.append(test_ll)
                        else:
                            f_test_costs.append(test_cost)
                            f_test_tms.append(test_tm)
                            f_test_lls.append(test_ll)
                        print('>>> TEST ({:.1f}s)'.format(time_test))
                        print('>> Test Cost = {:.8f}'.format(test_cost))
                        if(task == "regression"):
                            print('>> Test RMSE = {:.8f}'.format(test_tm))
                            print('>> Test NLPD = {:.8f}'.format(test_ll))
                        elif(task == "classification"):
                            print('>> Test Err Rate = {:.8f}'.format(test_tm))
                            print('>> Test AUC = {:.8f}'.format(test_ll))
                        if(not flag_cvrg):
                            cnt_cvrg = 0
                        if(best_cost < train_cost):
                            cnt_cvrg += 1
                        else:
                            best_epoch = len(f_train_costs)
                            cnt_cvrg = 0
                            best_cost = train_cost
                            if(SAVE):
                                saver = tf.train.saver()
                                if not os.path.exists('./trained/'):
                                    os.makedirs('./trained/')
                                saver.SAVE(sess,
                                    './trained/'+model_code+'_'+problem_name)
                            else:
                                min_tm, max_ll = test_tm, test_ll
                        if(cnt_cvrg > EARLY_STOP-(epoch*EARLY_STOP/MAX_EPOCHS)):
                            break
                
                # Load the selected best params and evaluate its performance
                if(SAVE):
                    saver = tf.train.saver()
                    saver.restore(sess,
                        './trained/'+model_code+'_'+problem_name)
                    tms, lls = [], []
                    t_iters = int(np.floor(X_test.shape[0]/float(BATCH_SIZE)))
                    for t in range(t_iters):
                        if(t == t_iters-1):                   
                            X_batch = X_test[t*BATCH_SIZE:]
                            y_batch = y_test[t*BATCH_SIZE:]
                        else:
                            X_batch = X_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                            y_batch = y_test[t*BATCH_SIZE:(t+1)*BATCH_SIZE]
                        mse, ll = sess.run(
                            [rms_error, LL],
                            feed_dict={n_samples: TEST_SAMPLES,
                                X: X_batch, y: y_batch})
                        tms.append(mse);lls.append(ll)
                    test_tm, test_ll = np.mean(tms), np.mean(lls)
                else:
                    test_tm, test_ll = min_tm, max_ll
                print('>>> BEST TEST')
                if(task == "regression"):
                    print('>> Test RMSE = {:.8f}'.format(test_tm))
                    print('>> Test NLPD = {:.8f}'.format(test_ll))
                elif(task == "classification"):
                    print('>> Test Err Rate = {:.8f}'.format(test_tm))
                    print('>> Test NLPD = {:.8f}'.format(test_ll))
                eval_tms[model_name].append(test_tm)
                eval_lls[model_name].append(test_ll)
            if(PLOT):
                import matplotlib.pyplot as plt
                plt.figure()
                plt.subplot(3, 1, 1)        
                plt.title(model_code+" on "+dataset_name)
                test_MAX_EPOCHS = np.arange(len(f_train_costs))*CHECK_FREQ
                plt.semilogx(test_MAX_EPOCHS, f_train_costs, '--', label='Train')
                plt.semilogx(test_MAX_EPOCHS, f_test_costs, label='Test')
                plt.xlabel('Epoch')
                plt.ylabel('Min Obj {:.4f}'.format(test_cost))
                plt.subplot(3, 1, 2)
                plt.semilogx(test_MAX_EPOCHS, f_train_tms, '--', label='Train')
                plt.semilogx(test_MAX_EPOCHS, f_test_tms, label='Test')
                plt.legend(loc='lower left')
                plt.xlabel('Epoch')
                if(task == "regression"):
                    plt.ylabel('RMSE {:.4f}'.format(test_tm))
                elif(task == "classification"):
                    plt.ylabel('CERR {:.4f}'.format(test_tm))
                    plt.ylim([0, 1])
                plt.subplot(3, 1, 3)
                plt.semilogx(test_MAX_EPOCHS, f_train_lls, '--', label='Train')
                plt.semilogx(test_MAX_EPOCHS, f_test_lls, label='Test')
                plt.xlabel('Epoch')
                if(task == "regression"):
                    plt.ylabel('NLPD {:.4f}'.format(test_ll))
                elif(task == "classification"):
                    plt.ylabel('AUC {:.4f}'.format(test_ll))
                if not os.path.exists('./plots/'):
                    os.makedirs('./plots/')
                plt.savefig('./plots/'+model_code+'_'+problem_name+'.png')
                plt.close()
            train_costs.append(np.array(f_train_costs))
            train_tms.append(np.array(f_train_tms))
            train_lls.append(np.array(f_train_lls))
            test_costs.append(np.array(f_test_costs))
            test_tms.append(np.array(f_test_tms))
            test_lls.append(np.array(f_test_lls))
    return eval_tms, eval_lls
