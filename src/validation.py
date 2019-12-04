"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
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

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import model_analyzer
import csv
from tensorflow.contrib.training.python.training.evaluation import checkpoints_iterator
import pdb
from pprint import pprint
import utils
import time
import models.inception_resnet_v1 as network
import tensorflow.contrib.slim as slim
import json
import re
from shutil import copyfile
import math
tf.app.flags.DEFINE_float('open_ratio', 0.0, '')
tf.app.flags.DEFINE_boolean('reset_weights_in_new_locations', False, '')
tf.app.flags.DEFINE_boolean('share_only_task_1', False, '')
tf.app.flags.DEFINE_boolean('verbose', False, '')
tf.app.flags.DEFINE_boolean('print_mem', False, '')
tf.app.flags.DEFINE_boolean('print_mask_info', False, '')
tf.app.flags.DEFINE_boolean('eval_once', False, '')
tf.app.flags.DEFINE_integer('task_id', 1, '')
tf.app.flags.DEFINE_string('csv_file_path', '', '')
tf.app.flags.DEFINE_string('task_name', 'age', '')
tf.app.flags.DEFINE_float('filters_expand_ratio', 1.0, "ratio of new filter's depth to initial filter's depth after this expension")
tf.app.flags.DEFINE_list('history_filters_expand_ratios', [1.0], "historical ratios of previous filters depth to initial filter's depth"
                                                                      "(including the ratio of this expension)")

FLAGS = tf.app.flags.FLAGS

def main(_):

    if FLAGS.csv_file_path:
      if os.path.exists(FLAGS.csv_file_path) is False:
        csv_dir = FLAGS.csv_file_path.rsplit('/', 1)[0]        
        if os.path.exists(csv_dir) is False:
          os.makedirs(csv_dir)

        if FLAGS.task_name == 'chalearn/age':
          with open(FLAGS.csv_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Pruned rate', 'MAE', 'Acc', 'Epoch No.',
                             'Model size through inference (MB) (Shared part + task-specific part)', 
                             'Shared part (MB)',
                             'Task specific part (MB)',
                             'Whole masks (MB)',
                             'Task specific masks (MB)',
                             'Task specific batch norm vars (MB)',
                             'Task specific biases (MB)'])
        else:
          with open(FLAGS.csv_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Pruned rate', 'Acc', 'Epoch No.',
                             'Model size through inference (MB) (Shared part + task-specific part)', 
                             'Shared part (MB)',
                             'Task specific part (MB)',
                             'Whole masks (MB)',
                             'Task specific masks (MB)',
                             'Task specific batch norm vars (MB)',
                             'Task specific biases (MB)'])

    args, unparsed = parse_arguments(sys.argv[1:])
    FLAGS.filters_expand_ratio = math.sqrt(FLAGS.filters_expand_ratio)
    FLAGS.history_filters_expand_ratios = [math.sqrt(float(ratio)) for ratio in FLAGS.history_filters_expand_ratios]

    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            if 'emotion' in FLAGS.task_name or 'chalearn' in FLAGS.task_name:
              test_data_path = os.path.join(args.data_dir, 'val')
            else:
              test_data_path = os.path.join(args.data_dir, 'test')

            test_set = utils.get_dataset(test_data_path)

            # Get the paths for the corresponding images
            image_list, label_list = facenet.get_image_paths_and_labels(test_set)

            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            # Load the model
            if os.path.isdir(args.model):
              temp_record_file = os.path.join(args.model, 'temp_record.txt')
              checkpoint_file = os.path.join(args.model, 'checkpoint')

              if os.path.exists(temp_record_file) and os.path.exists(checkpoint_file):
                with open(temp_record_file) as json_file:
                  data = json.load(json_file)
                  max_acc = max(data, key=float)
                  epoch_no = data[max_acc]
                  ckpt_file = args.model + '/model-.ckpt-' + epoch_no

                with open(checkpoint_file) as f:
                  context = f.read()
                original_epoch = re.search("(\d)+", context).group()
                context = context.replace(original_epoch, epoch_no)
                with open(checkpoint_file, 'w') as f:
                  f.write(context)
                if os.path.exists(os.path.join(args.model, 'copied')) is False:
                  os.makedirs(os.path.join(args.model, 'copied'))
                copyfile(temp_record_file, os.path.join(args.model, 'copied', 'temp_record.txt'))
                os.remove(temp_record_file)

              elif os.path.exists(checkpoint_file):
                ckpt = tf.train.get_checkpoint_state(args.model)
                ckpt_file = ckpt.model_checkpoint_path
                epoch_no = ckpt_file.rsplit('-', 1)[-1]
              else:
                print('No `temp_record.txt` or `checkpoint` in `{}`, you should pass args.model the file path, not the directory'.format(args.model))
                sys.exit(1)
            else:
              ckpt_file = args.model
              epoch_no = ckpt_file.rsplit('-')[-1]

            prelogits, _ = network.inference(image_batch, 1.0, 
                phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
                weight_decay=0.0)

            with tf.variable_scope('task_{}'.format(FLAGS.task_id)):
              if FLAGS.task_name == 'chalearn/age':
                logits = slim.fully_connected(prelogits, 100, activation_fn=None, scope='Logits', reuse=False)
              else:
                logits = slim.fully_connected(prelogits, len(test_set), activation_fn=None, scope='Logits', reuse=False)

            # Get output tensor
            if FLAGS.task_name == 'chalearn/age':
              softmax = tf.nn.softmax(logits=logits)
              labels_range = tf.range(1.0, 101.0)  # [1.0, ..., 100.0]
              labels_matrix = tf.broadcast_to(labels_range, [args.test_batch_size, labels_range.shape[0]])
              result_vector = tf.reduce_sum(softmax * labels_matrix, axis=1) 
              MAE_error_vector = tf.abs(result_vector - tf.cast(label_batch, tf.float32))
              MAE_avg_error = tf.reduce_mean(MAE_error_vector)

              correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
              accuracy = tf.reduce_mean(correct_prediction)
              regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
              total_loss = tf.add_n([MAE_avg_error] + regularization_losses)

              criterion = MAE_avg_error
            else:
              cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=label_batch, logits=logits, name='cross_entropy_per_example')
              cross_entropy_mean = tf.reduce_mean(cross_entropy)
              
              correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
              accuracy = tf.reduce_mean(correct_prediction)
              regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
              total_loss = tf.add_n([cross_entropy_mean] + regularization_losses)

              criterion = cross_entropy_mean

            init_fn = slim.assign_from_checkpoint_fn(ckpt_file, tf.global_variables())
            init_fn(sess)

            pruned_ratio_relative_to_curr_task = 0.0
            model_size = 0.0
            if FLAGS.print_mem or FLAGS.print_mask_info:
                masks = tf.get_collection('masks')
              
                if FLAGS.print_mask_info:

                  if masks:
                    num_elems_in_each_task_op = {}
                    num_elems_in_tasks_in_masks_op = {}  # two dimentional dictionary 
                    num_elems_in_masks_op = []
                    num_remain_elems_in_masks_op = []

                    for task_id in range(1, FLAGS.task_id+1):
                      num_elems_in_each_task_op[task_id] = tf.constant(0, dtype=tf.int32)
                      num_elems_in_tasks_in_masks_op[task_id] = {}

                    # Define graph
                    for i, mask in enumerate(masks):
                      num_elems_in_masks_op.append(tf.size(mask))
                      num_remain_elems_in_curr_mask = tf.size(mask)
                      for task_id in range(1, FLAGS.task_id+1):
                        cnt = tf_count(mask, task_id)
                        num_elems_in_tasks_in_masks_op[task_id][i] = cnt
                        num_elems_in_each_task_op[task_id] = tf.add(num_elems_in_each_task_op[task_id], cnt)
                        num_remain_elems_in_curr_mask -= cnt

                      num_remain_elems_in_masks_op.append(num_remain_elems_in_curr_mask)

                    num_elems_in_network_op = tf.add_n(num_elems_in_masks_op)

                    print('Calculate pruning status ...')

                    # Doing operation
                    num_elems_in_masks = sess.run(num_elems_in_masks_op)
                    num_elems_in_each_task = sess.run(num_elems_in_each_task_op)
                    num_elems_in_tasks_in_masks = sess.run(num_elems_in_tasks_in_masks_op)
                    num_elems_in_network = sess.run(num_elems_in_network_op)
                    num_remain_elems_in_masks = sess.run(num_remain_elems_in_masks_op)

                    # Print out the result
                    print('Showing pruning status ...')

                    if FLAGS.verbose:
                      for i, mask in enumerate(masks):
                        print('Layer %s: ' % mask.op.name, end='')
                        for task_id in range(1, FLAGS.task_id+1):
                          cnt = num_elems_in_tasks_in_masks[task_id][i]
                          print('task_%d -> %d/%d (%.2f%%), ' % (task_id, cnt, num_elems_in_masks[i], 100 *  cnt / num_elems_in_masks[i]), end='')
                        print('remain -> {:.2f}%'.format(100 * num_remain_elems_in_masks[i] / num_elems_in_masks[i]))

                    print('Num elems in network: {}'.format(num_elems_in_network))
                    num_elems_of_usued_weights = num_elems_in_network 
                    for task_id in range(1, FLAGS.task_id+1):
                      print('Num elems in task_{}: {}'.format(task_id, num_elems_in_each_task[task_id]))
                      print('Ratio of task_{} to all: {}'.format(task_id, num_elems_in_each_task[task_id] / num_elems_in_network))
                      num_elems_of_usued_weights -= num_elems_in_each_task[task_id]
                    print('Num usued elems in all masks: {}'.format(num_elems_of_usued_weights))

                    pruned_ratio_relative_to_all_elems = num_elems_of_usued_weights / num_elems_in_network
                    print('Ratio of usused_elem to all: {}'.format(pruned_ratio_relative_to_all_elems))
                    pruned_ratio_relative_to_curr_task = num_elems_of_usued_weights / (num_elems_of_usued_weights + num_elems_in_each_task[FLAGS.task_id])
                    print('Pruning degree relative to task_{}: {:.3f}'.format(FLAGS.task_id, pruned_ratio_relative_to_curr_task))
                
                if FLAGS.print_mem:
                  # Analyze param
                  start_time = time.time()
                  (MB_of_model_through_inference, 
                   MB_of_shared_variables, 
                   MB_of_task_specific_variables,
                   MB_of_whole_masks,
                   MB_of_task_specific_masks, 
                   MB_of_task_specific_batch_norm_variables, 
                   MB_of_task_specific_biases) = model_analyzer.analyze_vars_for_current_task(tf.model_variables(), 
                      sess=sess, task_id=FLAGS.task_id, verbose=False)
                  duration = time.time() - start_time
                  print('duration time: {}'.format(duration))
            if FLAGS.eval_once:
                validate(args, sess, image_list, label_list, eval_enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                    phase_train_placeholder, batch_size_placeholder,
                    total_loss, regularization_losses, criterion, accuracy, args.use_fixed_image_standardization, 
                    FLAGS.csv_file_path, pruned_ratio_relative_to_curr_task, epoch_no,
                    MB_of_model_through_inference,
                    MB_of_shared_variables, 
                    MB_of_task_specific_variables,
                    MB_of_whole_masks,
                    MB_of_task_specific_masks, 
                    MB_of_task_specific_batch_norm_variables, 
                    MB_of_task_specific_biases)
                
            return

def validate(args, sess, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
             phase_train_placeholder, batch_size_placeholder, 
             loss, regularization_losses, criterion, accuracy, use_fixed_image_standardization,
             csv_file_path, pruned_ratio, epoch_no,
             MB_of_model_through_inference,
             MB_of_shared_variables,
             MB_of_task_specific_variables,
             MB_of_whole_masks,
             MB_of_task_specific_masks,
             MB_of_task_specific_batch_norm_variables,
             MB_of_task_specific_biases):
  
    print('Running forward pass on validation set')

    nrof_batches = len(label_list) // args.test_batch_size
    nrof_images = nrof_batches * args.test_batch_size
    
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
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.test_batch_size}
        loss_, criterion_, accuracy_ = sess.run([loss, criterion, accuracy], feed_dict=feed_dict)
        loss_array[i], xent_array[i], accuracy_array[i] = (loss_, criterion_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time
    acc_mean = np.mean(accuracy_array)

    if FLAGS.task_name == 'chalearn/age':
      MAE = np.mean(xent_array)
      print('Validation Time %.3f\tLoss %2.3f\tMAE %2.3f\tAccuracy %2.3f' %
            (duration, np.mean(loss_array), MAE, acc_mean))    
    else:
      print('Validation Time %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
            (duration, np.mean(loss_array), np.mean(xent_array), acc_mean))    

    if csv_file_path:
      with open(csv_file_path, 'a') as f:
        writer = csv.writer(f)

        if FLAGS.task_name == 'chalearn/age':
          writer.writerow([
              '{:.5f}'.format(pruned_ratio),
              '{:.5f}'.format(MAE), '{:.5f}'.format(acc_mean), epoch_no,
              '{:.3f}'.format(MB_of_model_through_inference), 
              '{:.3f}'.format(MB_of_shared_variables), 
              '{:.3f}'.format(MB_of_task_specific_variables),
              '{:.3f}'.format(MB_of_whole_masks),
              '{:.3f}'.format(MB_of_task_specific_masks), 
              '{:.3f}'.format(MB_of_task_specific_batch_norm_variables), 
              '{:.3f}'.format(MB_of_task_specific_biases)])
        else:
          writer.writerow([
              '{:.5f}'.format(pruned_ratio), 
              '{:.5f}'.format(acc_mean), epoch_no,
              '{:.3f}'.format(MB_of_model_through_inference), 
              '{:.3f}'.format(MB_of_shared_variables), 
              '{:.3f}'.format(MB_of_task_specific_variables),
              '{:.3f}'.format(MB_of_whole_masks),
              '{:.3f}'.format(MB_of_task_specific_masks), 
              '{:.3f}'.format(MB_of_task_specific_batch_norm_variables), 
              '{:.3f}'.format(MB_of_task_specific_biases)])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned test data.')
    parser.add_argument('--test_batch_size', type=int,
        help='Number of images to process in a batch in the test set.', default=100)
    parser.add_argument('--json_file', type=str,
        help='Path to json_file')
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=512)
    # return parser.parse_args(argv)
    return parser.parse_known_args(argv)

def tf_count(tensor, value):
  elements_equal_to_value = tf.equal(tensor, value)
  as_ints = tf.cast(elements_equal_to_value, tf.int32)
  count = tf.reduce_sum(as_ints)
  return count

if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    tf.app.run()
