# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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


# Revise the official slim code for custom use

"""Tools for analyzing the operations and variables in a TensorFlow graph.

To analyze the operations in a graph:

  images, labels = LoadData(...)
  predictions = MyModel(images)

  slim.model_analyzer.analyze_ops(tf.get_default_graph(), print_info=True)

To analyze the model variables in a graph:

  variables = tf.model_variables()
  slim.model_analyzer.analyze_vars(variables, print_info=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pdb
from pprint import pprint


def tensor_description(var):
  """Returns a compact and informative string about a tensor.

  Args:
    var: A tensor variable.

  Returns:
    a string with type and size, e.g.: (float32 1x8x8x1024).
  """
  description = '(' + str(var.dtype.name) + ' '
  sizes = var.get_shape()
  for i, size in enumerate(sizes):
    description += str(size)
    if i < len(sizes) - 1:
      description += 'x'
  description += ')'
  return description


def analyze_ops(graph, print_info=False):
  """Compute the estimated size of the ops.outputs in the graph.

  Args:
    graph: the graph containing the operations.
    print_info: Optional, if true print ops and their outputs.

  Returns:
    total size of the ops.outputs
  """
  if print_info:
    print('---------')
    print('Operations: name -> (type shapes) [size]')
    print('---------')
  total_size = 0
  for op in graph.get_operations():
    op_size = 0
    shapes = []
    for output in op.outputs:
      # if output.num_elements() is None or [] assume size 0.
      output_size = output.get_shape().num_elements() or 0
      if output.get_shape():
        shapes.append(tensor_description(output))
      op_size += output_size
    if print_info:
      print(op.name, '\t->', ', '.join(shapes), '[' + str(op_size) + ']')
    total_size += op_size
  return total_size


def analyze_vars(variables, print_info=False):
  """Prints the names and shapes of the variables.

  Args:
    variables: list of variables, for example tf.global_variables().
    print_info: Optional, if true print variables and their shape.

  Returns:
    (total size of the variables, total bytes of the variables)
  """
  if print_info:
    print('---------')
    print('Variables: name (type shape) [size]')
    print('---------')
  total_size = 0
  total_bytes = 0
  for var in variables:
    # if var.num_elements() is None or [] assume size 0.
    var_size = var.get_shape().num_elements() or 0
    var_bytes = var_size * var.dtype.size
    total_size += var_size
    total_bytes += var_bytes
    if print_info:
      print(var.name, tensor_description(var), '[%d, bytes: %d]' %
            (var_size, var_bytes))
  if print_info:
    print('Total size of variables: %d' % total_size)
    print('Total bytes of variables: %d' % total_bytes)
  return total_size, total_bytes

def analyze_vars_for_current_task(variables, sess, task_id=1, verbose=0):
  

  bytes_of_shared_variables_op = tf.constant(0, dtype=tf.int32) # assume only share task 1
  bytes_of_task_specific_variables_op = tf.constant(0, dtype=tf.int32) # including classifier layer's variables and the task-specific parts in the shared layers
  
  bytes_of_whole_masks_op = tf.constant(0, dtype=tf.int32)
  bytes_of_task_specific_masks_op = tf.constant(0, dtype=tf.int32)
  bytes_of_task_specific_batch_norm_variables_op = tf.constant(0, dtype=tf.int32) # calculate task-specific batch_norm size
  bytes_of_task_specific_biases_op = tf.constant(0, dtype=tf.int32) # calculate task-specific biases size


  var_size_list_op = []
  var_bytes_list_op = []

  task_str = 'task_{}'.format(task_id)
  # ex. var name starting by task_1, task_2 task_3, are not share, so split it out
  non_shared_layer_variables = []
  shared_layer_variables = []

  mask_pairs = {
    'mask': 'weights',
    'depthwise_mask': 'depthwise_weights',
    'pointwise_mask': 'pointwise_weights',
  }

  weights_pairs = {
    'weights': None,
    'depthwise_weights': None,
    'pointwise_weights': None,
  }
  
  
  for var in variables:
    if var.name.startswith(task_str):
      # task_specific layers' variables
      var_size = tf.constant(var.get_shape().num_elements() or 0, dtype=tf.int32)
      var_bytes = var_size * var.dtype.size
      var_size_list_op.append(var_size)
      var_bytes_list_op.append(var_bytes)
      
      bytes_of_task_specific_variables_op += var_bytes
      non_shared_layer_variables.append(var)
    else:
      # shared layers' variables -> including both shared variables and task_specific variables in shared layers
      shared_layer_variables.append(var)

  other_variables = []
  # shared variables (Assume share only task 1)

  for var in shared_layer_variables:
    # if var.num_elements() is None or [] assume size 0.
    key = var.op.name.rsplit('/', 1)[-1]

    var_size = tf.constant(var.get_shape().num_elements() or 0, dtype=tf.int32)
    var_bytes = var_size * var.dtype.size

    if key in mask_pairs:
      weights_pairs[mask_pairs[key]] = var
      shared_var_size = tf_count(var, 1)
      shared_var_bytes = shared_var_size * 1      
      task_specific_var_size = tf_count(var, task_id)
      task_specific_var_bytes = task_specific_var_size * 1
      var_bytes = var_size * 1

      bytes_of_shared_variables_op += shared_var_bytes
      bytes_of_task_specific_variables_op += task_specific_var_bytes
      bytes_of_whole_masks_op += var_bytes
      bytes_of_task_specific_masks_op += task_specific_var_bytes

    elif key in weights_pairs:
      shared_var_size = tf_count(weights_pairs[key], 1)
      shared_var_bytes = shared_var_size * var.dtype.size
      task_specific_var_size = tf_count(weights_pairs[key], task_id)
      task_specific_var_bytes = task_specific_var_size * var.dtype.size

      bytes_of_shared_variables_op += shared_var_bytes
      bytes_of_task_specific_variables_op += task_specific_var_bytes

    else:
      # BatchNorm layers, threshold_variables, biases      
      task_specific_var_size = var_size
      task_specific_var_bytes = var_bytes
      bytes_of_task_specific_variables_op += task_specific_var_bytes
      
      if 'BatchNorm' in var.op.name:
        bytes_of_task_specific_batch_norm_variables_op += task_specific_var_bytes
      elif key == 'biases':
        bytes_of_task_specific_biases_op += task_specific_var_bytes

    var_size_list_op.append(var_size)
    var_bytes_list_op.append(var_bytes)

  if verbose:
    var_size_list = sess.run(var_size_list_op)
    var_bytes_list = sess.run(var_bytes_list_op)

  # MB_of_shared_variables = sess.run(tf_bytes_to_MB(bytes_of_shared_variables_op))
  # MB_of_task_specific_variables = sess.run(tf_bytes_to_MB(bytes_of_task_specific_variables_op))
  # MB_of_whole_masks = sess.run(tf_bytes_to_MB(bytes_of_whole_masks_op))
  # MB_of_task_specific_masks = sess.run(tf_bytes_to_MB(bytes_of_task_specific_masks_op))
  # MB_of_task_specific_batch_norm_variables = sess.run(tf_bytes_to_MB(bytes_of_task_specific_batch_norm_variables_op))
  # MB_of_task_specific_biases = sess.run(tf_bytes_to_MB(bytes_of_task_specific_biases_op))

  # When sequentially calculating numerals, CPU faster than GPU
  MB_of_shared_variables = bytes_to_MB(sess.run(bytes_of_shared_variables_op))
  MB_of_task_specific_variables = bytes_to_MB(sess.run(bytes_of_task_specific_variables_op))
  MB_of_whole_masks = bytes_to_MB(sess.run(bytes_of_whole_masks_op))
  MB_of_task_specific_masks = bytes_to_MB(sess.run(bytes_of_task_specific_masks_op))
  MB_of_task_specific_batch_norm_variables = bytes_to_MB(sess.run(bytes_of_task_specific_batch_norm_variables_op))
  MB_of_task_specific_biases = bytes_to_MB(sess.run(bytes_of_task_specific_biases_op))

  MB_of_model_through_inference = MB_of_shared_variables + MB_of_task_specific_variables


  # Print stage:
  if verbose:
    print('---------')
    print('Variables: name (type shape) [size]')
    print('---------')    
    ordered_visited_variables = non_shared_layer_variables + shared_layer_variables
    for var, var_size, var_bytes in zip(ordered_visited_variables, var_size_list, var_bytes_list):      
      print(var, tensor_description(var), '[%d, bytes: %d]' %
            (var_size, var_bytes))
  print()
  print('Model size (Contain only the inferenced one) : {:.2f} MB'.format(MB_of_model_through_inference))
  print('Shared part (Task 1 features) : {:.3f} MB'.format(MB_of_shared_variables))
  print('Task_{}-specific part : {:.3f} MB'.format(task_id, MB_of_task_specific_variables))
  print('Whole mask : {:.3f} MB'.format(MB_of_whole_masks))
  print('Task_{} mask : {:.3f} MB'.format(task_id, MB_of_task_specific_masks))
  print('Task_{} batch_norm : {:.3f} MB'.format(task_id, MB_of_task_specific_batch_norm_variables))
  print('Task_{} biases : {:.3f} MB'.format(task_id, MB_of_task_specific_biases))
  print()
  return (MB_of_model_through_inference, MB_of_shared_variables, MB_of_task_specific_variables, MB_of_whole_masks, 
          MB_of_task_specific_masks, MB_of_task_specific_batch_norm_variables, MB_of_task_specific_biases)

def tf_count(tensor, value):
  elements_equal_to_value = tf.equal(tensor, value)
  as_ints = tf.cast(elements_equal_to_value, tf.int32)
  count = tf.reduce_sum(as_ints)
  return count

def tf_bytes_to_MB(tensor):
  return tf.cast(tensor, tf.float32) / (1024*1024)

def bytes_to_MB(python_int_scalar):
  return float(python_int_scalar) / (1024*1024)