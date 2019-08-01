
import csv
import argparse
import pandas as pd
import pdb
import subprocess
import os
import sys

# python src/read_old_csv_and_output_new_csv.py --model_dir /home/ivclab/fevemania/facenet/official_checkpoint/facenet/0 \
# --old_csv_path /home/ivclab/fevemania/facenet/csv/facenet/0.csv \
# --new_csv_path /home/ivclab/fevemania/facenet/csv/facenet/task_1.csv \
# --task_id 1 \
# --task_name facenet \
# --gpu_id 1

# python src/read_old_csv_and_output_new_csv.py \
# --model_dir /home/ivclab/fevemania/facenet/official_checkpoint/experiment1/age/test_fold_is_0_finetune \
# --old_csv_path /home/ivclab/fevemania/facenet/csv/experiment1/age/test_fold_is_0_finetune.csv \
# --new_csv_path /home/ivclab/fevemania/facenet/csv/experiment1/age/test_fold_is_0_finetune_new.csv \
# --task_id 1 \
# --task_name age/test_fold_is_0 \
# --gpu_id 0

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,
    help='Directory where to read trained checkpoints.', default='/home/ivclab/fevemania/facenet/official_checkpoint/experiment2/emotion/1')
parser.add_argument('--old_csv_path', type=str, default='/home/ivclab/fevemania/facenet/csv/experiment2/emotion/1.csv',
    help="csv file to store the history of last task's pruning ratio and its corresponding accuracy")
parser.add_argument('--new_csv_path', type=str, default='/home/ivclab/fevemania/facenet/csv/experiment2/emotion/task_2.csv')
parser.add_argument('--best_finetune_checkpoint_epoch', '--best_epoch', type=int, default=0)
parser.add_argument('--task_id', type=int, required=True)
parser.add_argument('--task_name', type=str, required=True)
parser.add_argument('--gpu_id', type=int, required=True)
args = parser.parse_args()

history = pd.read_csv(args.old_csv_path)

epoch_numbers = history['Epoch No.'].tolist()

fixed_pattern = 'model-.ckpt-{}'

# ckpt = os.path.join(args.model_dir, fixed_pattern.format(args.best_finetune_checkpoint_epoch))

if args.task_name != 'facenet':
  # subprocess.call('CUDA_VISIBLE_DEVICES={} python /home/ivclab/fevemania/facenet/src/validation.py'
  #     ' --data_dir ~/fevemania/datasets/{}'
  #     ' --model {}'
  #     ' --use_fixed_image_standardization'
  #     ' --task_name {}'
  #     ' --task_id {}'
  #     ' --eval_once'
  #     ' --print_mem'
  #     ' --print_mask_info'
  #     ' --csv_file_path {}'.format(args.gpu_id, args.task_name, ckpt, args.task_name, args.task_id, args.new_csv_path), shell=True)

  for epoch_no in epoch_numbers:
    ckpt = os.path.join(args.model_dir, fixed_pattern.format(epoch_no))
    subprocess.call('CUDA_VISIBLE_DEVICES={} python /home/ivclab/fevemania/facenet/src/validation.py'
        ' --data_dir ~/fevemania/datasets/{}'
        ' --model {}'
        ' --use_fixed_image_standardization'
        ' --task_name {}'
        ' --task_id {}'
        ' --eval_once'
        ' --print_mem'
        ' --print_mask_info'
        ' --csv_file_path {}'.format(args.gpu_id, args.task_name, ckpt, args.task_name, args.task_id, args.new_csv_path), shell=True)

elif args.task_id == 1:

  for epoch_no in epoch_numbers:
    ckpt = os.path.join(args.model_dir, fixed_pattern.format(epoch_no))
    subprocess.call('CUDA_VISIBLE_DEVICES={} python /home/ivclab/fevemania/facenet/src/validate_on_lfw.py'
        ' --lfw_dir ~/fevemania/datasets/lfw_mtcnnpy_160'
        ' --lfw_pairs data/pairs.txt'
        ' --distance_metric 1'
        ' --use_flipped_images'
        ' --subtract_mean'
        ' --use_fixed_image_standardization'
        ' --model {}'
        ' --task_name {}'
        ' --task_id {}'
        ' --eval_once'
        ' --print_mem'
        ' --print_mask_info'
        ' --csv_file_path {}'.format(args.gpu_id, ckpt, args.task_name, args.task_id, args.new_csv_path), shell=True)
else:
  print('task_name is facenet, but task_id != 1, Error')
  sys.exit(1)