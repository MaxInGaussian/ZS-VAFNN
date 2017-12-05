''' training_graph.py '''
import numpy as np
import tensorflow as tf
from SGPA_graph import build_SGPA_graph
from sklearn.neighbors import KernelDensity

## Computation graph for training
def build_training_graph(layers_width, n_basis, learn_rate, Y_std):
    
    n_samples = tf.placeholder(tf.int32, shape=())
    X = tf.placeholder(tf.float32, shape=[None, layers_width[0]])
    Y = tf.placeholder(tf.float32, shape=[None, layers_width[-1]])
    
    ## Computation graph for optimization
    F, KL = build_SGPA_graph(X, layers_width, n_samples, n_basis)
    # F_mean, F_variance = tf.nn.moments(F, axes=[0])
    F_mean = tf.reduce_mean(F, 0)
    F_variance = tf.reduce_mean((F-tf.expand_dims(F_mean, 0))**2, 0)
    # noise = tf.get_variable('noise', shape=(),
    #     initializer=tf.constant_initializer(0.2))
    F_variance += F_mean**2
    # obj = tf.log(noise)+tf.reduce_mean((Y-F_mean)**2)/noise+KL
    obj = tf.losses.mean_squared_error(F_mean, Y)
    global_step = tf.Variable(0, trainable=False)
    learn_rate_ts = tf.train.exponential_decay(
        learn_rate, global_step, 10000, 0.96, staircase=True)
    optimizer = tf.train.AdadeltaOptimizer(learn_rate_ts)
    infer_op = optimizer.minimize(obj, global_step=global_step)
    
    ## Computation graph for saving the best set of parameters
    save_vars = []
    for var in tf.trainable_variables():
        save_vars.append(tf.Variable(var.initialized_value()))
    assign_to_save, assign_to_restore = [], []
    for var, save_var in zip(tf.trainable_variables(), save_vars):
        assign_to_save.append(save_var.assign(var))
        assign_to_restore.append(var.assign(save_var))
    
    ## Computation graph for evaluation
    rmse = tf.sqrt(tf.losses.mean_squared_error(F_mean*Y_std, Y*Y_std))
    nlpd = .5*tf.reduce_mean(tf.log(F_variance*Y_std**2.)+\
        (F_mean-Y)**2/F_variance)+.5*np.log(2*np.pi)
    return locals()