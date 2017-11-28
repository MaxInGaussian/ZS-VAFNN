''' run_graphs.py '''
import numpy as np
import tensorflow as tf
from training_graph import build_training_graph
from load_data import load_data, scaler

valid_freq, patience, epochs = 5, 10, 1000
learn_rate, batch_size, hiddens_width = 1e-4, 20, [50, 25]
train_samples, valid_samples, test_samples = 5, 20, 200

performance_log, datasets = [], load_data(5)
for X_train, Y_train, X_valid, Y_valid, X_test, Y_test in datasets:
    X_train, X_valid, X_test, X_mean, X_std = scaler(X_train, X_valid, X_test)
    Y_train, Y_valid, Y_test, Y_mean, Y_std = scaler(Y_train, Y_valid, Y_test)
    (N, D), (M, P) = X_train.shape, Y_valid.shape
    layers_width = [D]+hiddens_width+[P]
    tf.reset_default_graph()
    graph = build_training_graph(layers_width, learn_rate, Y_std)
    with tf.Session() as sess:
        wait, min_obj = 0, np.Infinity
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            train_batches, valid_batches = [], []
            for i in range(N//batch_size):
                X_b = X_train[i*batch_size:(i+1)*batch_size]
                Y_b = Y_train[i*batch_size:(i+1)*batch_size]
                train_batches.append(sess.run([graph['infer_op'], graph['obj']],
                    feed_dict={graph['X']: X_b, graph['Y']: Y_b,
                        graph['n_samples']: train_samples})[1])
            train_obj = np.mean(train_batches)
            if(epoch % valid_freq):
                print('Epoch %d: Train Obj = %.5f'%(epoch, train_obj))
            else:
                for i in range(M//batch_size):
                    X_b = X_valid[i*batch_size:(i+1)*batch_size]
                    Y_b = Y_valid[i*batch_size:(i+1)*batch_size]
                    valid_batches.append(sess.run([graph['obj']],
                        feed_dict={graph['X']: X_b, graph['Y']: Y_b,
                            graph['n_samples']: valid_samples}))
                valid_obj = np.mean(valid_batches)
                if(valid_obj < min_obj):
                    wait, min_obj = 0, valid_obj
                    sess.run(graph['assign_to_save'])
                else:
                    wait = wait + 1
                    if(wait >= patience): break
                print('Epoch %d: Valid Obj = %.5f <-> Min Obj = %.5f'%(
                    epoch, valid_obj, min_obj))
        sess.run(graph['assign_to_restore'])
        performance_log.append(sess.run([graph['rmse'], graph['nlpd']],
            feed_dict={graph['X']: X_test, graph['Y']: Y_test,
                graph['n_samples']: test_samples}))
        print('New Test RMSE = {:.4f}'.format(performance_log[-1][0]))
        print('New Test NLPD = {:.4f}'.format(performance_log[-1][1]))
        mu = np.mean(performance_log, 0)
        std = np.std(performance_log, 0)
        print('Overall Test RMSE = {:.4f} \pm {:.4f}'.format(mu[0], 1.96*std[0]))
        print('Overall Test NLPD = {:.4f} \pm {:.4f}'.format(mu[1], 1.96*std[1]))