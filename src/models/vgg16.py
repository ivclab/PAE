from pruning_layers import masked_conv2d, masked_fully_connected

def vgg_16(inputs, label_count):  
    weight_decay = 5e-4
    batch_size = 100
    with tf.variable_scope("Conv1", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(inputs, 3, 64, 3)
    with tf.variable_scope("Conv2", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 64, 64, 3)
        current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv3", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 64, 128, 3)
    with tf.variable_scope("Conv4", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 128, 128, 3)
        current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv5", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 128, 256, 3)
    with tf.variable_scope("Conv6", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 256, 256, 3)
    with tf.variable_scope("Conv7", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 256, 256, 3)
        current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv8", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 256, 512, 3)
    with tf.variable_scope("Conv9", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 512, 512, 3)
    with tf.variable_scope("Conv10", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 512, 512, 3)
        current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv11", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 512, 512, 3)
    with tf.variable_scope("Conv12", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 512, 512, 3)
    with tf.variable_scope("Conv13", reuse = tf.AUTO_REUSE):
        current = masked_conv2d(current, 512, 512, 3)
        current = maxpool2d(current, k=2)
        current = tf.reshape(current, [ -1, 512 ])
    with tf.variable_scope("FC14", reuse = tf.AUTO_REUSE):
        current = fc_batch_activ(current, 512, 4096)
    with tf.variable_scope("FC15", reuse = tf.AUTO_REUSE):
        current = fc_batch_activ(current, 4096, 4096)
    with tf.variable_scope("FC16", reuse = tf.AUTO_REUSE):
        Wfc = weight_variable_xavier([ 4096, label_count ], name = 'W')
        bfc = bias_variable([ label_count ])
        ys_ = tf.matmul(current, Wfc) + bfc