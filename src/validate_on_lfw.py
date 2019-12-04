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
from tensorflow.python import debug as tf_debug
from PIL import Image
import models.inception_resnet_v1 as network
import tensorflow.contrib.slim as slim
import time
import json
import re
from shutil import copyfile
import math

tf.app.flags.DEFINE_float('open_ratio', 0.0, '')
tf.app.flags.DEFINE_boolean('reset_weights_in_new_locations', False, '')
tf.app.flags.DEFINE_boolean('share_only_task_1', True, '')
tf.app.flags.DEFINE_boolean('verbose', False, '')
tf.app.flags.DEFINE_boolean('print_mem', False, '')
tf.app.flags.DEFINE_boolean('print_mask_info', False, '')
tf.app.flags.DEFINE_boolean('eval_once', False, '')
tf.app.flags.DEFINE_integer('task_id', 1, '')
tf.app.flags.DEFINE_string('csv_file_path', '', '')
tf.app.flags.DEFINE_string('task_name', 'facenet', '')
tf.app.flags.DEFINE_float('model_size_expand_ratio', 1.0, "ratio of expanded model's size to the original model size of the given network")
tf.app.flags.DEFINE_list('history_model_size_expand_ratio', [1.0], "expanded ratios of all of the historical models' size that we have assigned")

tf.app.flags.DEFINE_float('filters_expand_ratio', 1.0, "ratio of new filter's depth to initial model filter's depth."
                                                       "This ratio is decided by the `model_size_expand_ratio` parameter")
tf.app.flags.DEFINE_list('history_filters_expand_ratios', [1.0], "historical ratios of previous filters depth to initial model filter's depth"
                                                                 "(including the ratio of this expension)"
                                                                 "This list is decided by the `history_model_size_expand_ratio` parameter")

FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.csv_file_path:
      if os.path.exists(FLAGS.csv_file_path) is False:
        csv_dir = FLAGS.csv_file_path.rsplit('/', 1)[0]
        if os.path.exists(csv_dir) is False:
          os.makedirs(csv_dir)
        with open(FLAGS.csv_file_path, 'w') as f:
          writer = csv.writer(f)
          writer.writerow(['Pruned rate', 'Acc Mean', 'Acc Std', 'Epoch No.',
                           'Model size through inference (MB) (Shared part + task-specific part)', 
                           'Shared part (MB)',
                           'Task specific part (MB)',
                           'Whole masks (MB)',
                           'Task specific masks (MB)',
                           'Task specific batch norm vars (MB)',
                           'Task specific biases (MB)'])

  
    args, unparsed = parse_arguments(sys.argv[1:])
    FLAGS.filters_expand_ratio = math.sqrt(FLAGS.model_size_expand_ratio)
    FLAGS.history_filters_expand_ratios = [math.sqrt(ratio) for ratio in FLAGS.history_model_size_expand_ratio]

    with tf.Graph().as_default():
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

            # img = Image.open(paths[0])

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

            # Cannot use meta graph, because we need to dynamically decide batch normalization in regard to current task_id
            # facenet.load_model(args.model, input_map=input_map) 
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            prelogits, _ = network.inference(image_batch, 1.0, 
                phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
                weight_decay=0.0)

            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

            init_fn = slim.assign_from_checkpoint_fn(ckpt_file, tf.global_variables())
            init_fn(sess)

            pruned_ratio = 0.0
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

                    # pruned_ratio = num_elems_of_usued_weights / num_elems_in_network
                    # print('Ratio of usused_elem to all: {}'.format(pruned_ratio))
                    # print('Pruning degree relative to task_{}: {:.3f}'.format(FLAGS.task_id, num_elems_of_usued_weights / (num_elems_of_usued_weights + num_elems_in_each_task[FLAGS.task_id])))
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
                evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
                    embeddings, label_batch, paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, args.distance_metric, args.subtract_mean,
                    args.use_flipped_images, args.use_fixed_image_standardization, FLAGS.csv_file_path, pruned_ratio, epoch_no,
                    MB_of_model_through_inference,
                    MB_of_shared_variables, 
                    MB_of_task_specific_variables,
                    MB_of_whole_masks,
                    MB_of_task_specific_masks, 
                    MB_of_task_specific_batch_norm_variables, 
                    MB_of_task_specific_biases)
            
            

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
             embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization,
             csv_file_path, pruned_ratio, epoch_no,
             MB_of_model_through_inference,
             MB_of_shared_variables,
             MB_of_task_specific_variables,
             MB_of_whole_masks,
             MB_of_task_specific_masks,
             MB_of_task_specific_batch_norm_variables,
             MB_of_task_specific_biases):
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
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    
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
    start_eval_time = time.time()
    # tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    tpr, fpr, accuracy = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    evaluate_time = time.time() - start_eval_time

    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    print('Accuracy: %2.5f+-%2.5f' % (acc_mean, acc_std))
    # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)

    print('Eval time: {}'.format(evaluate_time))
    
    if csv_file_path:
      with open(csv_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            '{:.5f}'.format(pruned_ratio), 
            '{:.5f}'.format(acc_mean), 
            '{:.5f}'.format(acc_std), epoch_no,
            '{:.3f}'.format(MB_of_model_through_inference), 
            '{:.3f}'.format(MB_of_shared_variables), 
            '{:.3f}'.format(MB_of_task_specific_variables),
            '{:.3f}'.format(MB_of_whole_masks),
            '{:.3f}'.format(MB_of_task_specific_masks), 
            '{:.3f}'.format(MB_of_task_specific_batch_norm_variables), 
            '{:.3f}'.format(MB_of_task_specific_biases)])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    # parser.add_argument('--model', type=str, 
       # help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--model', type=str,
         help='Checkpoint file, not path')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
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
