"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import math
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import pruning
import re
from tensorflow.python.ops.variable_scope import _get_default_variable_store
import pdb
import utils
from pprint import pprint
import json
import math
from models import inception_resnet_v1 as network

tf.app.flags.DEFINE_float('open_ratio', 0.0, '')
# tf.app.flags.DEFINE_boolean('set_zeros_in_masks_to_current_task_id', False, '')

tf.app.flags.DEFINE_boolean('reset_weights_in_new_locations', False, '')
tf.app.flags.DEFINE_boolean('share_only_task_1', False, '')
tf.app.flags.DEFINE_string('cell_scope_to_be_assigned_current_task_id', '', '')

tf.app.flags.DEFINE_boolean(
    'use_pruning_strategy', False, '')
tf.app.flags.DEFINE_integer('begin_pruning_epoch', 0, '')
tf.app.flags.DEFINE_float('end_pruning_epoch', 3.0, '') # epoch pruning epoch can be float
tf.app.flags.DEFINE_integer('task_id', 1, '')

tf.app.flags.DEFINE_string('pruning_hparams',
  'name=pruning,initial_sparsity=0.0,target_sparsity=0.2,pruning_frequency=10',
    help='Comma seperated list of pruning-related hyperparameters')

tf.app.flags.DEFINE_string('special_operation', '', 
    'The name of the special operation, one of "expand",'
    'or "finetune".')
tf.app.flags.DEFINE_boolean('inherit_prev_task_bn', False, '')
tf.app.flags.DEFINE_string('task_name', 'facenet', '')
tf.app.flags.DEFINE_string('change_weight_name_from_github', '', '')
tf.app.flags.DEFINE_integer('max_to_keep', 4, '')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'has_pretrained_model_for_curr_task', False, '')

tf.app.flags.DEFINE_float('model_size_expand_ratio', 1.0, "ratio of expanded model's size to the original model size of the given network")
tf.app.flags.DEFINE_list('history_model_size_expand_ratio', [1.0], "expanded ratios of all of the historical models' size that we have assigned")

tf.app.flags.DEFINE_float('filters_expand_ratio', 1.0, "ratio of new filter's depth to initial model filter's depth."
                                                       "This ratio is decided by the `model_size_expand_ratio` parameter")
tf.app.flags.DEFINE_list('history_filters_expand_ratios', [1.0], "historical ratios of previous filters depth to initial model filter's depth"
                                                                 "(including the ratio of this expension)"
                                                                 "This list is decided by the `history_model_size_expand_ratio` parameter")

FLAGS = tf.app.flags.FLAGS

# Assuming that 'conv1/task_1/bn' should be restore from 'conv1/bn'
def mapping_name_function(var):
    var_name = var.op.name
    if 'last_mask_update_step' not in var_name:
      var_name = var_name.replace('task_1/', '')
    return var_name

def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
    args, unparsed = parse_arguments(sys.argv[1:])
    
    FLAGS.filters_expand_ratio = math.sqrt(FLAGS.filters_expand_ratio)
    FLAGS.history_filters_expand_ratios = [math.sqrt(float(ratio)) for ratio in FLAGS.history_filters_expand_ratios]
    
    # network = importlib.import_module(args.model_def)
    image_size = (args.image_size, args.image_size)

    # subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    subdir = ''
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)

    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    if FLAGS.task_name == 'facenet':
        dataset = facenet.get_dataset(os.path.join(args.data_dir, 'train'))

        if args.filter_filename:
            dataset = filter_dataset(dataset, os.path.expanduser(args.filter_filename), 
                args.filter_percentile, args.filter_min_nrof_images_per_class)
            
        if args.validation_set_split_ratio>0.0:
            train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
        else:
            train_set, val_set = dataset, []
    else:
        train_data_path = os.path.join(args.data_dir, 'train')
        validation_data_path = os.path.join(args.data_dir, 'val')
        train_set = utils.get_dataset(train_data_path)
        val_set = utils.get_dataset(validation_data_path)

        if FLAGS.task_name == 'chalearn/age':
          pass
        else:
          assert len(train_set) == len(val_set)

    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    if FLAGS.has_pretrained_model_for_curr_task:
        json_file = os.path.join(args.models_base_dir, 'pretrained_file_for_curr_task.txt')
        if not os.path.exists(json_file):
          print("Sorry, you don't have pretrained file history.")
          sys.exit(1)
        else:
          with open(json_file) as f:
            pretrained_model = json.load(f)
            print('Pre-trained model: %s' % pretrained_model)
            
    if FLAGS.task_name == 'facenet':
      if args.lfw_dir:
          print('LFW directory: %s' % args.lfw_dir)
          # Read the file containing the pairs used for testing
          pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
          # Get the paths for the corresponding images
          lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
      else:
          print('Please provide LFW dir, Error!')
          sys.exit(1)

    task_str = 'task_{}'.format(FLAGS.task_id)

    with tf.Graph().as_default():
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        tf.set_random_seed(args.seed)
        with tf.variable_scope(task_str):
            global_step = slim.create_global_step() #tf.Variable(0, trainable=False)
        
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list)>0, 'The training set should not be empty'
        
        num_training_examples = len(image_list)
        args.epoch_size = min(num_training_examples // args.batch_size, args.epoch_size)
        val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
        
        nrof_preprocess_threads = 4
        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        image_batch, label_batch = facenet.create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))
        print('Epoch size: {}'.format(args.epoch_size))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))
        
        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)

        with tf.variable_scope('task_{}'.format(FLAGS.task_id)):
          
          logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
                  weights_initializer=slim.initializers.xavier_initializer(), 
                  weights_regularizer=slim.l2_regularizer(args.weight_decay),
                  scope='Logits', reuse=False)

        if FLAGS.task_name == 'facenet':
          embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

          # Norm for the prelogits
          eps = 1e-4
          prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=args.prelogits_norm_p, axis=1))
          tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

          # Add center loss
          prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
          tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)
        else:
          prelogits_norm = tf.no_op()
          prelogits_center_loss = tf.no_op()


        # learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            # args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        # tf.summary.scalar('learning_rate', learning_rate)

        if 'emotion' in FLAGS.task_name:
          print('here')
          onehot_labels = tf.one_hot(label_batch, nrof_classes)
          class_counts = np.array([74496, 133758, 25311, 14016, 6322, 3783, 24726])
          min_count = class_counts.min()
          total_count = class_counts.sum()
          class_weights = np.ones(nrof_classes)
          for i in range(nrof_classes):
            class_weights[i] = (total_count - class_counts[i]) / class_counts[i]

          class_w = tf.convert_to_tensor(class_weights, tf.float32)
          class_w = tf.reshape(class_w, [1, nrof_classes])
          weight_per_label = tf.transpose(tf.matmul(onehot_labels, tf.transpose(class_w)))
          cross_entropy = tf.multiply(weight_per_label, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits, name='cross_entropy_per_example'))
        else:
          # Calculate the average cross entropy loss across the batch
          cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=label_batch, logits=logits, name='cross_entropy_per_example')

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        
        if FLAGS.task_name == 'chalearn/age':
          softmax = tf.nn.softmax(logits=logits)
          labels_range = tf.range(1.0, 101.0)  # [1.0, ..., 100.0]
          labels_matrix = tf.broadcast_to(labels_range, [args.batch_size, labels_range.shape[0]])
          result_vector = tf.reduce_sum(softmax * labels_matrix, axis=1) 
          MAE_error_vector = tf.abs(result_vector - tf.cast(label_batch, tf.float32))
          MAE_avg_error = tf.reduce_mean(MAE_error_vector)

          tf.add_to_collection('losses', MAE_avg_error)
          # correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
          # accuracy = tf.reduce_mean(correct_prediction)

          # Calculate the total losses
          total_loss = tf.add_n([cross_entropy_mean, MAE_avg_error] + regularization_losses, name='total_loss')
          criterion = MAE_avg_error

        else:
          criterion = cross_entropy_mean        

          # Calculate the total losses
          total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        ckpt = tf.train.get_checkpoint_state(model_dir)
        restart_epoch = 0
        if ckpt and ckpt.model_checkpoint_path:
            restart_epoch = re.search(r'model-.ckpt-([0-9]*)', ckpt.model_checkpoint_path)
            if restart_epoch != None:
                restart_epoch = int(restart_epoch.group(1))

        # Pruning Setting
        if FLAGS.use_pruning_strategy:
          num_batches_per_epoch = args.epoch_size
          print('Start configuring pruning')
          pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)
          pruning_hparams.begin_pruning_step = FLAGS.begin_pruning_epoch * num_batches_per_epoch
          pruning_hparams.end_pruning_step = int(FLAGS.end_pruning_epoch * num_batches_per_epoch)
          restart_step = restart_epoch * num_batches_per_epoch
          pruning_hparams.begin_pruning_step += restart_step
          pruning_hparams.end_pruning_step += restart_step
          pruning_hparams.sparsity_function_begin_step = pruning_hparams.begin_pruning_step 
          pruning_hparams.sparsity_function_end_step = pruning_hparams.end_pruning_step
          print('pruning stage between step {} and {}'.format(pruning_hparams.begin_pruning_step, pruning_hparams.end_pruning_step))
          print('pruning ratio from {} to {}'.format(pruning_hparams.initial_sparsity, pruning_hparams.target_sparsity))
          pruning_obj = pruning.Pruning(FLAGS.task_id, pruning_hparams, global_step=tf.train.get_global_step())
          # Use the pruning object to add ops to the training graph to update the masks
          mask_update_op = pruning_obj.conditional_mask_update_op()
          print('Finish configuring pruning')
        else:
          mask_update_op = tf.no_op()
          print('Disable pruning strategy')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        # train_op = facenet.train(total_loss, global_step, args.optimizer, 
            # learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)

        variables_to_train = _get_variables_to_train()
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate_placeholder, args.moving_average_decay, variables_to_train, args.log_histograms)

        ###########################################
        ## Change checkpoint weights' name #
        ###########################################

        if FLAGS.change_weight_name_from_github:
          if not tf.gfile.IsDirectory(model_dir):
            print('model_dir must be path to store the converted pretrain weights')
            return
          
          variables_to_restore = tf.global_variables()
          
          exclusions = []

          if FLAGS.checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

          selected_variables_to_restore = []
          
          for var in variables_to_restore:
            for exclusion in exclusions:
              if exclusion in var.op.name:
                break
            else:
              selected_variables_to_restore.append(var)

          selected_variables_to_restore = {
              mapping_name_function(var):var
              for var in selected_variables_to_restore}


          init_fn = slim.assign_from_checkpoint_fn(FLAGS.change_weight_name_from_github, selected_variables_to_restore, ignore_missing_vars=True)
          
          # filename = FLAGS.change_weight_name_from_github.rsplit('/')[-1]
          # metagraph_filename = os.path.join(model_dir, 'model-.meta')

          saver = tf.train.Saver(selected_variables_to_restore)
          saver2 = tf.train.Saver()
          with tf.Session() as sess:
              sess.run(tf.global_variables_initializer())
              sess.run(tf.local_variables_initializer())
              init_fn(sess)
              coord = tf.train.Coordinator()
              tf.train.start_queue_runners(coord=coord, sess=sess)
              epoch = 0
              step = 0
              if FLAGS.task_name == 'facenet':
                summary_writer = None
                
                evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, 
                    embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer, epoch, 
                    args.lfw_distance_metric, args.lfw_subtract_mean, args.lfw_use_flipped_images, args.use_fixed_image_standardization)              
              else:
                validate(args, sess, epoch, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                        phase_train_placeholder, batch_size_placeholder, total_loss, regularization_losses, criterion, accuracy, 
                        args.validate_every_n_epochs, args.use_fixed_image_standardization)

              saver2.save(sess, '{}/model-.ckpt-0'.format(model_dir), write_meta_graph=False)
              # saver2.export_meta_graph(metagraph_filename)
          return

        ###########################################
        ## Restore Architecture of previous tasks #
        ###########################################

        _ps = _private_store_space = _get_default_variable_store()

        saving_variables = tf.model_variables()
        
        additional_variables_to_be_saved = []
        for var in tf.global_variables():
          for key_name in ['step', 'Adam', 'beta1_power', 'beta2_power']:
            if key_name in var.name:
              additional_variables_to_be_saved.append(var)

        saving_variables += additional_variables_to_be_saved


        if pretrained_model or (ckpt and ckpt.model_checkpoint_path):
          if pretrained_model:
            print("Find pretrained_model, import previous tasks' variables!")
            step = re.search(r'model-.ckpt-([0-9]*)', pretrained_model)
            if step != None:
              step = step.group(1)
            previous_checkpoint_file = pretrained_model
          else:
            print("Find checkpoint, import previous tasks' variables!")
            step = re.search(r'model-.ckpt-([0-9]*)', ckpt.model_checkpoint_path).group(1)
            previous_checkpoint_file = ckpt.model_checkpoint_path

          if step != None:
            assign_value_list = []

            history_task_strs = ['task_{}'.format(i) for i in range(1, FLAGS.task_id+1)]
            if history_task_strs:
              task_str_to_be_inherited_from = history_task_strs[-1]      
              if FLAGS.share_only_task_1:
                task_str_to_be_inherited_from = 'task_1'
            else:
              task_str_to_be_inherited_from = None

            for var_name, _ in tf.contrib.framework.list_variables(previous_checkpoint_file):
              value = tf.contrib.framework.load_variable(previous_checkpoint_file, var_name)

              for key_name in ['Adam', 'ExponentialMovingAverage', 'beta1_power', 'beta2_power', 'cross_entropy/avg', 'total_loss/avg', 'RMSProp', 'Momentum']:
                if key_name in var_name:
                  # try:
                  #   var = tf.get_variable(name=var_name + '/', shape=value.shape, dtype=value.dtype)
                  #   assign_value_list.append(var.assign(value))
                  # except:
                  #   # already have this node in graph and with new shape
                  #   # Need to align the checkpoint
                  #   old_shape = value.shape
                  #   new_shape = _ps._vars[var_name + '/'].shape
                  #   idx = [slice(0, min(a, b)) for (a, b) in zip(old_shape, new_shape)]
                  #   var = _ps._vars[var_name + '/']
                  #   assign_value_list.append(var[idx].assign(value))
                  break
              else:
                try:
                  idx = [slice(0, dim) for dim in zip(value.shape)]

                  if FLAGS.special_operation == 'expand' and ('BatchNorm' in var_name or 'biases' in var_name) and 'Logits' not in var_name:
                    for old_str in history_task_strs:
                      if old_str in var_name:
                        old_shape = value.shape
                        new_shape = _ps._vars[var_name.replace(old_str, task_str)].shape
                        idx = [slice(0, min(a, b)) for (a, b) in zip(old_shape, new_shape)]
                        var = tf.get_variable(name=var_name, shape=new_shape, dtype=value.dtype)
                        assign_value_list.append(var[idx].assign(value))
                        if var not in saving_variables:
                          saving_variables.append(var)
                        break
                  else:
                    var = tf.get_variable(name=var_name, shape=value.shape, dtype=value.dtype)
                    assign_value_list.append(var.assign(value))
                    if var not in saving_variables:
                      saving_variables.append(var)

                  # inherit bn layer with/without expand
                  # if FLAGS.inherit_prev_task_bn and ('bn' in var_name) and task_str_to_be_inherited_from and (task_str_to_be_inherited_from in var_name) and ('aux_7' not in var_name):
                  #   var = tf.get_variable(name=var_name.replace(task_str_to_be_inherited_from, task_str))
                  #   assign_value_list.append(var[idx].assign(value))
                  
                except:
                  # already have this node in graph and with new shape (or new dtype)
                  # First checkpoint if it is fine-tune operation
                  if FLAGS.special_operation == 'finetune':
                    continue

                  if FLAGS.special_operation != 'expand':
                    print(" Did not specify `expand` operation, however, some variables' shapes in checkpoint cannot match the correspond variables in current graph!")
                    pdb.set_trace()
                    sys.exit(1)

                  # Masks or weight is going to expand              
                  old_shape = value.shape
                  new_shape = _ps._vars[var_name].shape
                  idx = [slice(0, min(a, b)) for (a, b) in zip(old_shape, new_shape)]
                  var = _ps._vars[var_name]
                  assign_value_list.append(var[idx].assign(value))
                  if var not in saving_variables:
                      saving_variables.append(var)

            print('Finish loading variables...')
          else:
            print("pretrained file or checkpoint doesn't match pattern, skip loading previous tasks' variables")
        
        # Create a saver
        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        saver = tf.train.Saver(saving_variables, max_to_keep=FLAGS.max_to_keep)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

          # nrof_steps = max_nrof_epochs*args.epoch_size
          # nrof_val_samples = int(math.ceil(max_nrof_epochs / args.validate_every_n_epochs))   # Validate every validate_every_n_epochs as well as in the last epoch
          if FLAGS.special_operation == 'expand':
            if step is not None:
              if not tf.gfile.Exists(model_dir):
                  tf.gfile.MakeDirs(model_dir)

              print("`network is expanded`, we need to adjust the checkpoint")        
              sess.run(tf.global_variables_initializer())
              sess.run(assign_value_list)
              epoch = sess.run(global_step, feed_dict=None)
              step = epoch
          elif pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)              
            init_fn = slim.assign_from_checkpoint_fn(pretrained_model, saving_variables, ignore_missing_vars=True)
            init_fn(sess)
            epoch = 0
            step = 0
          elif ckpt and ckpt.model_checkpoint_path:
            print('Restoring checkpoint file in model dir: %s' % ckpt.model_checkpoint_path)
            init_fn = slim.assign_from_checkpoint_fn(ckpt.model_checkpoint_path, saving_variables, ignore_missing_vars=True)
            init_fn(sess)
            step = int(step)
            epoch = step

          if FLAGS.open_ratio != 0.0:
            if FLAGS.task_name == 'facenet':
              evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, 
                  embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer, epoch, 
                  args.lfw_distance_metric, args.lfw_subtract_mean, args.lfw_use_flipped_images, args.use_fixed_image_standardization)
            else:
              validate(args, sess, epoch, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                      phase_train_placeholder, batch_size_placeholder, 
                      total_loss, regularization_losses, criterion, accuracy, args.validate_every_n_epochs, args.use_fixed_image_standardization)

            save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)
            tf.logging.info("finish saving")
            return              

          print('Start global_step: {}'.format(sess.run(global_step, feed_dict=None)))
          # saver.restore(sess, pretrained_model)

          # Training and validation loop
          print('Running training')

          start_epoch = 1
          if restart_epoch != 0:
            start_epoch += restart_epoch        
          max_nrof_epochs = start_epoch + args.nrof_addiitonal_epochs_to_run 

          if FLAGS.use_pruning_strategy:
            if FLAGS.task_name == 'chalearn/age':
              curr_best_criterion, curr_best_epoch = float('inf'), 0  # when pruning, we don't care tracking curr_best_criterion and curr_best_epoch
            else:
              curr_best_criterion, curr_best_epoch = 0.0, 0  # when pruning, we don't care tracking curr_best_criterion and curr_best_epoch

          elif FLAGS.task_name == 'facenet':
            epoch = 0
            step = 0
            curr_best_criterion, curr_best_epoch = evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, 
                        embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer, epoch, 
                        args.lfw_distance_metric, args.lfw_subtract_mean, args.lfw_use_flipped_images, args.use_fixed_image_standardization)
          elif FLAGS.task_name == 'chalearn/age':
            curr_best_criterion, curr_best_epoch = validate(args, sess, start_epoch-1, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                phase_train_placeholder, batch_size_placeholder, 
                total_loss, regularization_losses, criterion, accuracy, args.validate_every_n_epochs, args.use_fixed_image_standardization, curr_best_criterion=float('inf'))
          else:
            curr_best_criterion, curr_best_epoch = validate(args, sess, start_epoch-1, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                phase_train_placeholder, batch_size_placeholder, 
                total_loss, regularization_losses, criterion, accuracy, args.validate_every_n_epochs, args.use_fixed_image_standardization)


          past_best_criterion = curr_best_criterion
          learning_rate = args.learning_rate
          epoch = start_epoch
          num_epochs_that_criterion_does_not_get_better = 0
          times_of_decaying_learning_rate_that_criterion_does_not_get_better = 0

          while epoch < max_nrof_epochs:
            step = sess.run(global_step, feed_dict=None)
            # Train for one epoch
            cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, global_step, 
                total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file,
                criterion, accuracy, learning_rate,
                prelogits, prelogits_center_loss, args.random_rotate, args.random_crop, args.random_flip, prelogits_norm, args.prelogits_hist_max, args.use_fixed_image_standardization,
                mask_update_op)
            
            # if not cont:
            #     break
              
            if FLAGS.task_name == 'facenet':
              curr_best_criterion, curr_best_epoch = evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, 
                  embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer, epoch, 
                  args.lfw_distance_metric, args.lfw_subtract_mean, args.lfw_use_flipped_images, args.use_fixed_image_standardization,
                  curr_best_criterion, curr_best_epoch)
            else:
              # if len(val_image_list)>0 and ((epoch-1) % args.validate_every_n_epochs == args.validate_every_n_epochs-1 or epoch==max_nrof_epochs-1):
              curr_best_criterion, curr_best_epoch = validate(
                  args, sess, epoch, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                  phase_train_placeholder, batch_size_placeholder, 
                  total_loss, regularization_losses, criterion, accuracy, args.validate_every_n_epochs, args.use_fixed_image_standardization,
                  curr_best_criterion, curr_best_epoch)

            if FLAGS.use_pruning_strategy:
              if curr_best_criterion != past_best_criterion:
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, curr_best_epoch)
                past_best_criterion = curr_best_criterion
            else:
              # Save variables and the metagraph if it doesn't exist already
              if curr_best_criterion != past_best_criterion:
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, curr_best_epoch)
                past_best_criterion = curr_best_criterion
                num_epochs_that_criterion_does_not_get_better = 0
                times_of_decaying_learning_rate_that_criterion_does_not_get_better = 0
              else:
                num_epochs_that_criterion_does_not_get_better += 1

              if num_epochs_that_criterion_does_not_get_better >= 15:
                times_of_decaying_learning_rate_that_criterion_does_not_get_better += 1
                if times_of_decaying_learning_rate_that_criterion_does_not_get_better == 2:
                  print("times_of_decaying_learning_rate_that_criterion_does_not_get_better reach {}, stop training".format(times_of_decaying_learning_rate_that_criterion_does_not_get_better))
                  sys.exit(0)

                learning_rate *= 0.1
                print("continously {} epochs doesn't get higher acc, decay learning rate by multiplying 0.1".format(num_epochs_that_criterion_does_not_get_better))
                # store to the previous best checkpoint before decaying learning rate
                restore_path = os.path.join(model_dir, 'model-.ckpt-{}'.format(curr_best_epoch))
                print('resotre previous best checkpoint from {}'.format(restore_path))
                init_fn = slim.assign_from_checkpoint_fn(restore_path, saving_variables, ignore_missing_vars=False)
                init_fn(sess)
                epoch = curr_best_epoch
                num_epochs_that_criterion_does_not_get_better = 0

            epoch += 1

    return model_dir
  
def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset
  
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step, 
      loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file, 
      criterion, accuracy, 
      learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm, prelogits_hist_max, use_fixed_image_standardization,
      mask_update_op):
    batch_number = 0
    
    lr = learning_rate

    if lr<=0:
        return False 

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, criterion, prelogits_norm, accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, criterion_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, criterion_, prelogits_norm_, accuracy_, center_loss_ = sess.run(tensor_list, feed_dict=feed_dict)
         
        duration = time.time() - start_time
        
        sess.run(mask_update_op)

        duration = time.time() - start_time
        if FLAGS.task_name == 'facenet':
          print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
                (epoch, batch_number+1, args.epoch_size, duration, loss_, criterion_, np.sum(reg_losses_), accuracy_, lr, center_loss_))
        elif FLAGS.task_name == 'chalearn/age':
          print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tMAE %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f' %
                (epoch, batch_number+1, args.epoch_size, duration, loss_, criterion_, np.sum(reg_losses_), accuracy_, lr))
        else:
          print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f' %
                (epoch, batch_number+1, args.epoch_size, duration, loss_, criterion_, np.sum(reg_losses_), accuracy_, lr))

        batch_number += 1
        train_time += duration

    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True

def validate(args, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
             phase_train_placeholder, batch_size_placeholder, 
             loss, regularization_losses, criterion, accuracy, validate_every_n_epochs, use_fixed_image_standardization,
             curr_best_criterion=0.0, curr_best_epoch=0):
  
    print('Running forward pass on validation set')

    nrof_batches = len(label_list) // args.batch_size
    nrof_images = nrof_batches * args.batch_size
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_list[:nrof_images]),1)
    image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]),1)
    control_array = np.ones_like(labels_array, np.int32)*facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    loss_array = np.zeros((nrof_batches,), np.float32)
    xent_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.batch_size}
        loss_, criterion_, accuracy_ = sess.run([loss, criterion, accuracy], feed_dict=feed_dict)
        loss_array[i], xent_array[i], accuracy_array[i] = (loss_, criterion_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time

    val_index = (epoch-1)//validate_every_n_epochs


    acc_mean = np.mean(accuracy_array)

    if FLAGS.task_name == 'chalearn/age':
      MAE = np.mean(xent_array)
      curr_best_MAE = curr_best_criterion
      print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tMAE %2.3f\tAccuracy %2.3f' %
            (epoch, duration, np.mean(loss_array), MAE, acc_mean))
    else:
      curr_best_acc = curr_best_criterion
      print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
            (epoch, duration, np.mean(loss_array), np.mean(xent_array), acc_mean))

    new_better_record_when_pruning = False
  

    if FLAGS.task_name == 'chalearn/age':
      if MAE < curr_best_MAE:
        new_better_record_when_pruning = True
        curr_best_MAE = MAE
        curr_best_epoch = epoch

      if FLAGS.use_pruning_strategy:
        if new_better_record_when_pruning:
          data = {}
          data['{:.5f}'.format(MAE)] = str(epoch)
          with open(os.path.join(args.models_base_dir, 'temp_record.txt'), 'w') as outfile:
            json.dump(data, outfile)

      print('current best MAE is {}; its epoch no is {}'.format(curr_best_MAE, curr_best_epoch))
      return curr_best_MAE, curr_best_epoch

    else:    
      if acc_mean > curr_best_acc:
        new_better_record_when_pruning = True
        curr_best_acc = acc_mean
        curr_best_epoch = epoch

      if FLAGS.use_pruning_strategy:
        if new_better_record_when_pruning:
          data = {}
          data['{:.5f}'.format(acc_mean)] = str(epoch)
          with open(os.path.join(args.models_base_dir, 'temp_record.txt'), 'w') as outfile:
            json.dump(data, outfile)

      print('current best acc is {}; its epoch no is {}'.format(curr_best_acc, curr_best_epoch))
      return curr_best_acc, curr_best_epoch


def evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, 
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer, epoch, distance_metric, subtract_mean, 
        use_flipped_images, use_fixed_image_standardization, curr_best_acc=0.0, curr_best_epoch=0):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    # _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    _, _, accuracy = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    acc_mean = np.mean(accuracy)

    print('Accuracy: %2.5f+-%2.5f' % (acc_mean, np.std(accuracy)))
    #print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time

    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=acc_mean)
    #summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    if summary_writer != None:
      summary_writer.add_summary(summary, step)


    new_better_record_when_pruning = False
    if acc_mean > curr_best_acc:
      new_better_record_when_pruning = True
      curr_best_acc = acc_mean
      curr_best_epoch = epoch

    if FLAGS.use_pruning_strategy:
      if new_better_record_when_pruning:
        data = {}
        data['{:.5f}'.format(acc_mean)] = str(epoch)
        with open(os.path.join(args.models_base_dir, 'temp_record.txt'), 'w') as outfile:
          json.dump(data, outfile)

    print('current best acc is {}; its epoch no is {}'.format(curr_best_acc, curr_best_epoch))
    return curr_best_acc, curr_best_epoch

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--nrof_addiitonal_epochs_to_run', type=int,
        help='Number of additional epochs to run.', default=100)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--lfw_subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')

    return parser.parse_known_args(argv)
  

if __name__ == '__main__':
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  tf.app.run()
  
