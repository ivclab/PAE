import csv
import argparse
import pandas as pd
import pdb
from pprint import pprint
import os
import json

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--prev_task_folder_name', type=str, default='facenet/0')
FLAGS.add_argument('--curr_task_folder_name', type=str, default='experiment1/age/test_fold_is_0')
FLAGS.add_argument('--curr_task_id', type=int, default=2)
args = FLAGS.parse_args()

prev_task_csv_path = os.path.join('/home/ivclab/fevemania/prac_DL/facenet/csv', args.prev_task_folder_name) + '.csv'
prev_task_ckpt_folder = os.path.join('/home/ivclab/fevemania/prac_DL/facenet/official_checkpoint', args.prev_task_folder_name)
curr_task_ckpt_folder = os.path.join('/home/ivclab/fevemania/prac_DL/facenet/official_checkpoint', args.curr_task_folder_name)
os.makedirs(curr_task_ckpt_folder, exist_ok=True)

history = pd.read_csv(prev_task_csv_path)
fixed_pattern = 'model.ckpt-{}'

if args.curr_task_id == 2:
  best_epoch_number_after_pruning = int(history.loc[history['Acc Mean'].idxmax()]['Epoch No.'])
else:
  best_epoch_number_after_pruning = int(history.loc[history['Acc'].idxmax()]['Epoch No.'])

pretrained_file_path_for_curr_task = os.path.join(prev_task_ckpt_folder, fixed_pattern.format(best_epoch_number_after_pruning))
with open(os.path.join(curr_task_ckpt_folder, 'pretrained_file_for_curr_task.txt'), 'w') as outfile:
  json.dump(pretrained_file_path_for_curr_task, outfile)
