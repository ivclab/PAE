# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from pruning_layers import masked_conv2d, masked_fully_connected
import pdb

FLAGS = tf.app.flags.FLAGS

def custom_concat_func_for_expand(states_to_concat, original_filter_numbers, axis=3):
  
  assert len(states_to_concat) == len(original_filter_numbers)

  if FLAGS.history_filters_expand_ratios:
    pivot_positions = [0] * len(original_filter_numbers)
    new_pivot_positions = []
    new_states_to_concat = []
    
    for prev_ratio in FLAGS.history_filters_expand_ratios:
      splits = []
      for state, original_filter_number, pivot_position in zip(states_to_concat, original_filter_numbers, pivot_positions):
        prev_num_conv_filter = int(original_filter_number * prev_ratio)
        splits.append(state[:, :, :, pivot_position:prev_num_conv_filter])
        new_pivot_positions.append(prev_num_conv_filter)

      temp = tf.concat(values=splits, axis=axis)
      new_states_to_concat.append(temp)
      pivot_positions = new_pivot_positions
      new_pivot_positions = []

    return tf.concat(new_states_to_concat, axis)
  else:
    return tf.concat(states_to_concat, axis)

# Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = masked_conv2d(net, int(32 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_1x1', task_id=FLAGS.task_id)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = masked_conv2d(net, int(32 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
            tower_conv1_1 = masked_conv2d(tower_conv1_0, int(32 * FLAGS.filters_expand_ratio), 3, scope='Conv2d_0b_3x3', task_id=FLAGS.task_id)
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = masked_conv2d(net, int(32 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
            tower_conv2_1 = masked_conv2d(tower_conv2_0, int(32 * FLAGS.filters_expand_ratio), 3, scope='Conv2d_0b_3x3', task_id=FLAGS.task_id)
            tower_conv2_2 = masked_conv2d(tower_conv2_1, int(32 * FLAGS.filters_expand_ratio), 3, scope='Conv2d_0c_3x3', task_id=FLAGS.task_id)
        
        states_to_concat = [tower_conv, tower_conv1_1, tower_conv2_2]

        mixed = custom_concat_func_for_expand(states_to_concat, [32, 32, 32], 3)
        up = masked_conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1', task_id=FLAGS.task_id)
        
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = masked_conv2d(net, int(128 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_1x1', task_id=FLAGS.task_id)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = masked_conv2d(net, int(128 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
            tower_conv1_1 = masked_conv2d(tower_conv1_0, int(128 * FLAGS.filters_expand_ratio), [1, 7],
                                        scope='Conv2d_0b_1x7', task_id=FLAGS.task_id)
            tower_conv1_2 = masked_conv2d(tower_conv1_1, int(128 * FLAGS.filters_expand_ratio), [7, 1],
                                        scope='Conv2d_0c_7x1', task_id=FLAGS.task_id)
        mixed = custom_concat_func_for_expand([tower_conv, tower_conv1_2], [128, 128], 3)
        up = masked_conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1', task_id=FLAGS.task_id)
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = masked_conv2d(net, int(192 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_1x1', task_id=FLAGS.task_id)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = masked_conv2d(net, int(192 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
            tower_conv1_1 = masked_conv2d(tower_conv1_0, int(192 * FLAGS.filters_expand_ratio), [1, 3],
                                        scope='Conv2d_0b_1x3', task_id=FLAGS.task_id)
            tower_conv1_2 = masked_conv2d(tower_conv1_1, int(192 * FLAGS.filters_expand_ratio), [3, 1],
                                        scope='Conv2d_0c_3x1', task_id=FLAGS.task_id)
        mixed = custom_concat_func_for_expand([tower_conv, tower_conv1_2], [192, 192], 3)
        up = masked_conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1', task_id=FLAGS.task_id)
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
  
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = masked_conv2d(net, int(n * FLAGS.filters_expand_ratio), 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3', task_id=FLAGS.task_id)
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = masked_conv2d(net, int(k * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
        tower_conv1_1 = masked_conv2d(tower_conv1_0, int(l * FLAGS.filters_expand_ratio), 3,
                                    scope='Conv2d_0b_3x3', task_id=FLAGS.task_id)
        tower_conv1_2 = masked_conv2d(tower_conv1_1, int(m * FLAGS.filters_expand_ratio), 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3', task_id=FLAGS.task_id)
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = custom_concat_func_for_expand([tower_conv, tower_conv1_2, tower_pool], [n, m, 256], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = masked_conv2d(net, int(256 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
        tower_conv_1 = masked_conv2d(tower_conv, int(384 * FLAGS.filters_expand_ratio), 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3', task_id=FLAGS.task_id)
    with tf.variable_scope('Branch_1'):
        tower_conv1 = masked_conv2d(net, int(256 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
        tower_conv1_1 = masked_conv2d(tower_conv1, int(256 * FLAGS.filters_expand_ratio), 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3', task_id=FLAGS.task_id)
    with tf.variable_scope('Branch_2'):
        tower_conv2 = masked_conv2d(net, int(256 * FLAGS.filters_expand_ratio), 1, scope='Conv2d_0a_1x1', task_id=FLAGS.task_id)
        tower_conv2_1 = masked_conv2d(tower_conv2, int(256 * FLAGS.filters_expand_ratio), 3,
                                    scope='Conv2d_0b_3x3', task_id=FLAGS.task_id)
        tower_conv2_2 = masked_conv2d(tower_conv2_1, int(256 * FLAGS.filters_expand_ratio), 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3', task_id=FLAGS.task_id)
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    
    net = custom_concat_func_for_expand([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], [384, 256, 256, 896],3)
    
    return net
  
def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    with slim.arg_scope([masked_conv2d, masked_fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None, 
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([masked_conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = masked_conv2d(inputs, int(32 * FLAGS.filters_expand_ratio), 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3', task_id=FLAGS.task_id)
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = masked_conv2d(net, int(32 * FLAGS.filters_expand_ratio), 3, padding='VALID',
                                  scope='Conv2d_2a_3x3', task_id=FLAGS.task_id)
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = masked_conv2d(net, int(64 * FLAGS.filters_expand_ratio), 3, scope='Conv2d_2b_3x3', task_id=FLAGS.task_id)
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = masked_conv2d(net, int(80 * FLAGS.filters_expand_ratio), 1, padding='VALID',
                                  scope='Conv2d_3b_1x1', task_id=FLAGS.task_id)
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = masked_conv2d(net, int(192 * FLAGS.filters_expand_ratio), 3, padding='VALID',
                                  scope='Conv2d_4a_3x3', task_id=FLAGS.task_id)
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = masked_conv2d(net, int(256 * FLAGS.filters_expand_ratio), 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3', task_id=FLAGS.task_id)
                end_points['Conv2d_4b_3x3'] = net
                
                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net
        
                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net
                
                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net
                
                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net
                
                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net
                
                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net
                
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
          
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
          
                    end_points['PreLogitsFlatten'] = net
                
                net = masked_fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False, task_id=FLAGS.task_id)
                
    return net, end_points
