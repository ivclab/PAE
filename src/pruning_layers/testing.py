import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MASKED_WEIGHT_NAME = 'weights/masked_weight'

a = tf.constant([1,2])
b = tf.constant([2,3])
c = tf.constant([3,4])
result1 = tf.multiply(a, b, MASKED_WEIGHT_NAME)
result2 = tf.multiply(a, c, MASKED_WEIGHT_NAME)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  print(sess.run(result1))
  print(sess.run(result2))

