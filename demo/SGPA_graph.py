''' SGPA_graph.py '''
import tensorflow as tf

## Computation graph for SGPA
def build_SGPA_graph(X, layers_width, n_samples):
    KL = 0
    Z = tf.expand_dims(tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1]), 2)
    for h, n_out in enumerate(layers_width[1:]):
        # Hidden layer
        if(h < len(layers_width)-2):
            # Perform affine mapping at each layer of the neural network
            Z = tf.layers.dense(Z, n_out)
            # Define variational parameters
            alpha_mean = tf.get_variable('alpha_mean_layer'+str(h),
                shape=[1, 1, n_out*2, n_out],
                initializer=tf.random_normal_initializer())
            alpha_logstd = tf.get_variable('alpha_logstd_layer'+str(h),
                shape=[1, 1, n_out*2, n_out],
                initializer=tf.random_normal_initializer())
            alpha_std = tf.exp(alpha_logstd)
            # Compute epsilon from {n_samples} standard Gaussian
            # epsilon = tf.random_normal([n_samples, 1, n_out*2, n_out])
            epsilon = tf.random_uniform([n_samples, 1, n_out*2, n_out])
            hyp_params = tf.get_variable('hyp_params_layer'+str(h),
                shape=[2],
                initializer=tf.random_normal_initializer())
            l1, l2 = tf.nn.sigmoid(hyp_params[0]), tf.exp(hyp_params[1])
            epsilon = tf.sinh(epsilon*l2)/tf.cosh(epsilon*l2)**l1/l2
            # Compute A_{h+1}
            A = tf.tile(alpha_mean+epsilon*alpha_std, [1, tf.shape(X)[0], 1, 1])
            # Compute z_{h}A_{h+1}
            Z1 = tf.matmul(Z, A[:,:,:n_out,:])/tf.sqrt(n_out*1.)
            Z2 = tf.matmul(Z, A[:,:,n_out:,:])/tf.sqrt(n_out*1.)
            # Compute u_{h+1} and v_{h+1}
            U, V = tf.cos(Z1)+tf.cos(Z2), tf.sin(Z1)+tf.sin(Z2)
            Z = tf.concat([U, V], 3)/tf.sqrt(n_out*1.)
            KL += tf.reduce_mean(alpha_std**2+alpha_mean**2-2*alpha_logstd-1)/2.
        # Output layer
        else:
            F = tf.squeeze(tf.layers.dense(Z, n_out), [2])
    return F, KL