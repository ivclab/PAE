# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the core layer classes for model pruning and its functional aliases.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import control_flow_ops
import pdb

MASK_COLLECTION = 'masks'
THRESHOLD_COLLECTION = 'thresholds'
MASKED_WEIGHT_COLLECTION = 'masked_weights'
WEIGHT_COLLECTION = 'kernel'
# The 'weights' part of the name is needed for the quantization library
# to recognize that the kernel should be quantized.
MASKED_WEIGHT_NAME = 'weights/masked_weight'

FLAGS = tf.app.flags.FLAGS

def change_mask(mask, task_id=1, open_ratio=0.2, cell_scope_name=''):
  
  if cell_scope_name not in mask.op.name:
    return control_flow_ops.no_op()

  # from 0 to 1, use setting probability to decide whether open the rest space to train
  # with tf.device('/cpu:0'):
  mask_in_gpu = tf.cast(mask, dtype=tf.int32) # GPU only support int32, int64, not int8
  task_id_mask = tf.constant(value=task_id, shape=mask_in_gpu.shape, dtype=tf.int32)  # GPU only support int32, int64, not int8
  false_mask = tf.constant(value=False, shape=mask_in_gpu.shape, dtype=tf.bool)

  with tf.device('/cpu:0'):
    random_open_mask = tf.random_uniform(shape=mask.shape, minval=0.0, maxval=1.0, dtype=tf.float32)  # if we don't put this op to cpu, error occurs!   
    select_mask = tf.where(tf.equal(mask, 0), tf.less(random_open_mask, open_ratio), false_mask)

  if task_id == 1:
    return tf.cast(mask.assign(tf.constant(value=1, shape=mask.shape, dtype=tf.int8)), dtype=tf.bool)
  else:
    return tf.cast(mask.assign(tf.cast(tf.where(select_mask, task_id_mask, mask_in_gpu), dtype=tf.int8)), dtype=tf.bool) # select part of elems in mask and change mask value 0.0 to new task_id    

def change_mask_and_weight(mask, weight, task_id=1, open_ratio=0.2, cell_scope_name=''):
  
  if cell_scope_name not in mask.op.name:
    return control_flow_ops.no_op()
  # from 0 to 1, use setting probability to decide whether open the rest space to train
  # with tf.device('/cpu:0'):
  mask_in_gpu = tf.cast(mask, dtype=tf.int32) # GPU only support int32, int64, not int8
  task_id_mask = tf.constant(value=task_id, shape=mask_in_gpu.shape, dtype=tf.int32)  # GPU only support int32, int64, not int8
  false_mask = tf.constant(value=False, shape=mask_in_gpu.shape, dtype=tf.bool)
  random_weight = tf.random_normal(shape=weight.shape, stddev=0.01, dtype=tf.float32)

  with tf.device('/cpu:0'):
    random_open_mask = tf.random_uniform(shape=mask.shape, minval=0.0, maxval=1.0, dtype=tf.float32)  # if we don't put this op to cpu, error occurs!   
    select_mask = tf.where(tf.equal(mask, 0), tf.less(random_open_mask, open_ratio), false_mask)

  if task_id == 1:
    return tf.cast(mask.assign(tf.constant(value=1, shape=mask.shape, dtype=tf.int8)), dtype=tf.bool)
  else:
    return tf.logical_and(
        tf.cast(mask.assign(tf.cast(tf.where(select_mask, task_id_mask, mask_in_gpu), dtype=tf.int8)), dtype=tf.bool), # select part of elems in mask and change mask value 0.0 to new task_id
        tf.cast(weight.assign(tf.where(select_mask, random_weight, weight)), dtype=tf.bool) # assign these part of elems in mask with new task_id the new random weights
    )

class _MaskedConv(base.Layer):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. The weight tensor of this layer is masked.
  If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               task_id = 1,
               **kwargs):
    super(_MaskedConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = utils.normalize_tuple(strides, rank, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(dilation_rate, rank,
                                               'dilation_rate')
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.input_spec = base.InputSpec(ndim=self.rank + 2)
    self.task_id = task_id

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = 1 if self.data_format == 'channels_first' else -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)
    self.mask = self.add_variable(
        name='mask',
        shape=kernel_shape,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=tf.int8)

    self.kernel = self.add_variable(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        trainable=True,
        dtype=self.dtype)

    self.threshold = self.add_variable(
        name='threshold',
        shape=[],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=self.dtype)

    if FLAGS.reset_weights_in_new_locations and FLAGS.open_ratio:
      conditional_op = change_mask_and_weight(self.mask, self.kernel, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)
    elif FLAGS.open_ratio:
      conditional_op = change_mask(self.mask, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)      
    else:
      conditional_op = control_flow_ops.no_op()
      
    # conditional_op = control_flow_ops.cond(
    #   manually_set_zeros_to_task_id(),
    #   lambda: control_flow_ops.no_op(),
    #   lambda: change_mask_and_weight(self.mask, self.kernel, self.task_id, FLAGS.open_ratio))

    # Add masked_weights in the weights namescope so as to make it easier
    # for the quantization library to add quant ops.
    

    with tf.control_dependencies([conditional_op]):
      if FLAGS.share_only_task_1:
        boolean_mask = tf.cast(
            tf.logical_or(
                tf.equal(tf.identity(self.mask), 1),
                tf.equal(tf.identity(self.mask), self.task_id)),
            dtype=tf.float32)
      else:
        boolean_mask = tf.cast(
            tf.logical_and(
                tf.greater_equal(tf.identity(self.mask), 1),
                tf.less_equal(tf.identity(self.mask), self.task_id)),
            dtype=tf.float32)

      self.masked_kernel = math_ops.multiply(boolean_mask, self.kernel,
                                             MASKED_WEIGHT_NAME)

      if self.mask not in ops.get_collection_ref(MASK_COLLECTION):
        ops.add_to_collection(MASK_COLLECTION, self.mask)
        ops.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_kernel)
        ops.add_to_collection(THRESHOLD_COLLECTION, self.threshold)
        ops.add_to_collection(WEIGHT_COLLECTION, self.kernel)

      if self.use_bias:
        original_scope = self._scope
        with tf.variable_scope('task_{}'.format(self.task_id)) as scope: # Because there are multi-task problems
          self._scope = scope
          self.bias = self.add_variable(
              name='bias',
              shape=(self.filters,),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              trainable=True,
              dtype=self.dtype)
        self._scope = original_scope
      else:
        self.bias = None

      self.input_spec = base.InputSpec(
          ndim=self.rank + 2, axes={channel_axis: input_dim})
      self.built = True

  def call(self, inputs):
    outputs = nn.convolution(
        input=inputs,
        filter=self.masked_kernel,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, self.rank + 2))

    if self.bias is not None:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        if self.rank == 2:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          outputs_4d = array_ops.reshape(outputs, [
              outputs_shape[0], outputs_shape[1],
              outputs_shape[2] * outputs_shape[3], outputs_shape[4]
          ])
          outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)

class MaskedSeparableConv2D(_MaskedConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               depth_multiplier=1, # only separable conv has
               activation=None,
               use_bias=True,
               depthwise_initializer='global_uniform', # only separable conv has
               pointwise_initializer='global_uniform', # only separable conv has
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_initializer=init_ops.zeros_initializer(),
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               task_id = 1,
               **kwargs):
    super(MaskedSeparableConv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        task_id=task_id,
        **kwargs)
    
    # only seperate conv have
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = depthwise_initializer
    self.pointwise_initializer = pointwise_initializer
    self.depthwise_regularizer = depthwise_regularizer
    self.pointwise_regularizer = pointwise_regularizer

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = 1 if self.data_format == 'channels_first' else -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value

    depthwise_kernel_shape = self.kernel_size + (input_dim,
                                                 self.depth_multiplier)
    pointwise_kernel_shape = (
        1,) * self.rank + (self.depth_multiplier * input_dim, self.filters)

    self.depthwise_mask = self.add_variable(
        name='depthwise_mask',
        shape=depthwise_kernel_shape,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=tf.int8)

    self.depthwise_kernel = self.add_variable(
        name='depthwise_kernel',
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        regularizer=self.depthwise_regularizer,
        trainable=True,
        dtype=self.dtype)

    self.depthwise_threshold = self.add_variable(
        name='depthwise_threshold',
        shape=[],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=self.dtype)

    self.pointwise_mask = self.add_variable(
        name='pointwise_mask',
        shape=pointwise_kernel_shape,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=tf.int8)

    self.pointwise_kernel = self.add_variable(
        name='pointwise_kernel',
        shape=pointwise_kernel_shape,
        initializer=self.pointwise_initializer,
        regularizer=self.pointwise_regularizer,
        trainable=True,
        dtype=self.dtype)

    self.pointwise_threshold = self.add_variable(
        name='pointwise_threshold',
        shape=[],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=self.dtype)

    if FLAGS.reset_weights_in_new_locations and FLAGS.open_ratio:
      depthwise_conditional_op = change_mask_and_weight(self.depthwise_mask, self.depthwise_kernel, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)
    elif FLAGS.open_ratio:
      depthwise_conditional_op = change_mask(self.depthwise_mask, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)
    else:
      depthwise_conditional_op = control_flow_ops.no_op()

    # depthwise_conditional_op = control_flow_ops.cond(
    #   manually_set_zeros_to_task_id(),
    #   lambda: control_flow_ops.no_op(),
    #   lambda: change_mask_and_weight(self.depthwise_mask, 
    #       self.depthwise_kernel, self.task_id, FLAGS.open_ratio))
    if FLAGS.reset_weights_in_new_locations and FLAGS.open_ratio:
      pointwise_conditional_op = change_mask_and_weight(self.pointwise_mask, self.pointwise_kernel, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)
    elif FLAGS.open_ratio:
      pointwise_conditional_op = change_mask(self.pointwise_mask, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)
    else:
      pointwise_conditional_op = control_flow_ops.no_op()    

    # pointwise_conditional_op = control_flow_ops.cond(
    #   manually_set_zeros_to_task_id(),
    #   lambda: control_flow_ops.no_op(),
    #   lambda: change_mask_and_weight(self.pointwise_mask,
    #       self.pointwise_kernel, self.task_id, FLAGS.open_ratio))

    # Add masked_weights in the weights namescope so as to make it easier
    # for the quantization library to add quant ops.
    with tf.control_dependencies([depthwise_conditional_op,
                                  pointwise_conditional_op]):
      if FLAGS.share_only_task_1:
        depthwise_boolean_mask = tf.cast(
            tf.logical_or(
                tf.equal(tf.identity(self.depthwise_mask), 1),
                tf.equal(tf.identity(self.depthwise_mask), self.task_id)),
            dtype=tf.float32)

        pointwise_boolean_mask = tf.cast(
            tf.logical_or(
                tf.equal(tf.identity(self.pointwise_mask), 1),
                tf.equal(tf.identity(self.pointwise_mask), self.task_id)),
            dtype=tf.float32)        
      else:
        depthwise_boolean_mask = tf.cast(
            tf.logical_and(
                tf.greater_equal(tf.identity(self.depthwise_mask), 1),
                tf.less_equal(tf.identity(self.depthwise_mask), self.task_id)),
            dtype=tf.float32)

        pointwise_boolean_mask = tf.cast(
            tf.logical_and(
                tf.greater_equal(tf.identity(self.pointwise_mask), 1),
                tf.less_equal(tf.identity(self.pointwise_mask), self.task_id)),
            dtype=tf.float32)

      self.masked_depthwise_kernel = math_ops.multiply(depthwise_boolean_mask, 
          self.depthwise_kernel,
          MASKED_WEIGHT_NAME)
      
      self.masked_pointwise_kernel = math_ops.multiply(pointwise_boolean_mask,
          self.pointwise_kernel,
          MASKED_WEIGHT_NAME)

      if self.depthwise_mask not in ops.get_collection_ref(MASK_COLLECTION):
        ops.add_to_collection(MASK_COLLECTION, self.depthwise_mask)
        ops.add_to_collection(MASK_COLLECTION, self.pointwise_mask)
        ops.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_depthwise_kernel)
        ops.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_pointwise_kernel)
        ops.add_to_collection(THRESHOLD_COLLECTION, self.depthwise_threshold)
        ops.add_to_collection(THRESHOLD_COLLECTION, self.pointwise_threshold)      
        ops.add_to_collection(WEIGHT_COLLECTION, self.depthwise_kernel)
        ops.add_to_collection(WEIGHT_COLLECTION, self.pointwise_kernel)

      if self.use_bias:
        original_scope = self._scope
        with tf.variable_scope('task_{}'.format(self.task_id)) as scope: # Because there are multi-task problems
          self._scope = scope
          self.bias = self.add_variable(
              name='bias',
              shape=(self.filters,),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              trainable=True,
              dtype=self.dtype)
        self._scope = original_scope          
      else:
        self.bias = None
      self.input_spec = base.InputSpec(
          ndim=self.rank + 2, axes={channel_axis: input_dim})
      self.built = True

  def call(self, inputs):
    # Apply the actual ops.
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides + (1,)
    else:
      strides = (1, 1) + self.strides

    outputs = nn.separable_conv2d(
        inputs,
        self.masked_depthwise_kernel,
        self.masked_pointwise_kernel,
        strides=strides,
        padding=self.padding.upper(),
        rate=self.dilation_rate,
        data_format=utils.convert_data_format(self.data_format, ndim=4))

    if self.bias is not None:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        if self.rank == 2:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          outputs_4d = array_ops.reshape(outputs, [
              outputs_shape[0], outputs_shape[1],
              outputs_shape[2] * outputs_shape[3], outputs_shape[4]
          ])
          outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

class MaskedConv2D(_MaskedConv):
  """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               task_id=1,
               **kwargs):
    super(MaskedConv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        task_id=task_id,
        **kwargs)


class MaskedFullyConnected(base.Layer):
  """Fully-connected layer class with masked weights.

  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.

  Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the weight matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the weight matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               task_id=1,
               **kwargs):
    super(MaskedFullyConnected, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.input_spec = base.InputSpec(min_ndim=2)
    self.task_id = task_id

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(
        min_ndim=2, axes={-1: input_shape[-1].value})

    self.mask = self.add_variable(
        name='mask',
        shape=[input_shape[-1].value, self.units],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=tf.int8)

    self.kernel = self.add_variable(
        'kernel',
        shape=[input_shape[-1].value, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        dtype=self.dtype,
        trainable=True)

    self.threshold = self.add_variable(
        name='threshold',
        shape=[],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=self.dtype)

    if FLAGS.reset_weights_in_new_locations and FLAGS.open_ratio:
      conditional_op = change_mask_and_weight(self.mask, self.kernel, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)
    elif FLAGS.open_ratio:
      conditional_op = change_mask(self.mask, self.task_id, FLAGS.open_ratio, FLAGS.cell_scope_to_be_assigned_current_task_id)
    else:
      conditional_op = control_flow_ops.no_op()  

    # conditional_op = control_flow_ops.cond(
    #   manually_set_zeros_to_task_id(),
    #   lambda: control_flow_ops.no_op(),
    #   lambda: change_mask_and_weight(self.mask, self.kernel, self.task_id, FLAGS.open_ratio))

    # Add masked_weights in the weights namescope so as to make it easier
    # for the quantization library to add quant ops.
    with tf.control_dependencies([conditional_op]):
      if FLAGS.share_only_task_1:
        boolean_mask = tf.cast(
            tf.logical_or(
                tf.equal(tf.identity(self.mask), 1),
                tf.equal(tf.identity(self.mask), self.task_id)),
            dtype=tf.float32)
      else:
        boolean_mask = tf.cast(
            tf.logical_and(
                tf.greater_equal(tf.identity(self.mask), 1),
                tf.less_equal(tf.identity(self.mask), self.task_id)),
            dtype=tf.float32)
      
      self.masked_kernel = math_ops.multiply(boolean_mask, self.kernel,
                                             MASKED_WEIGHT_NAME)

      if self.mask not in ops.get_collection_ref(MASK_COLLECTION):
        ops.add_to_collection(MASK_COLLECTION, self.mask)
        ops.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_kernel)
        ops.add_to_collection(THRESHOLD_COLLECTION, self.threshold)
        ops.add_to_collection(WEIGHT_COLLECTION, self.kernel)


      if self.use_bias:
        original_scope = self._scope
        with tf.variable_scope('task_{}'.format(self.task_id)) as scope: # Because there are multi-task problems
          self._scope = scope
          self.bias = self.add_variable(
              name='bias',
              shape=(self.filters,),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              trainable=True,
              dtype=self.dtype)
        self._scope = original_scope
      else:
        self.bias = None
      self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    output_shape = shape[:-1] + [self.units]
    if len(output_shape) > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.masked_kernel,
                                       [[len(shape) - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      outputs.set_shape(output_shape)
    else:
      outputs = standard_ops.matmul(inputs, self.masked_kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)
