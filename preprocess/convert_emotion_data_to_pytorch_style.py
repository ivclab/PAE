import pandas as pd
import os
import sys
from PIL import Image
from shutil import copyfile
import csv
from tqdm import tqdm
import argparse
import pdb

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--train_csv_path', type=str, default='emotion/training.csv',
    help='csv file that contains data_paths which point to training images')
FLAGS.add_argument('--raw_data_path', type=str, default='/home/ivclab/fevemania/datasets/emotion/align_raw_182/0',
    help='folder that contains training images')
FLAGS.add_argument('--train_path', type=str, default='/home/ivclab/fevemania/datasets/emotion/train',
    help='folder where train images save in order to make pytorch ImageFolder class can read')
FLAGS.add_argument('--csv_file_to_store_failed_train_image_paths', type=str, default='/home/ivclab/fevemania/datasets/emotion/ignore_train_y.csv',
    help='csv file to store failed train image paths that are caused by mtcnn algorithm')
args = FLAGS.parse_args()

train_data_info = pd.read_csv(args.train_csv_path)

train_x = train_data_info['subDirectory_filePath'].tolist()
train_y = train_data_info['expression'].tolist()
num_origin_train_y = len(train_y)
classes = list(set(train_y))
num_classes = len(classes)


num_ignore_train_y = 0
for class_id in classes:
  path = os.path.join(args.train_path, str(class_id))
  if os.path.isdir(path):
    pass
  else:
    os.makedirs(path)

if os.path.isfile(args.csv_file_to_store_failed_train_image_paths) is False:
  csv_dir = args.csv_file_to_store_failed_train_image_paths.rsplit('/', 1)[0]
  if os.path.exists(csv_dir) is False:
    os.makedirs(csv_dir)

  with open(args.csv_file_to_store_failed_train_image_paths, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Failed file path'])
else:
  print('already tackle converted to pytorch style train image data')
  sys.exit(0)

with open(args.csv_file_to_store_failed_train_image_paths, 'w') as f:
  writer = csv.writer(f)
  for old_image_path, class_id in tqdm(zip(train_x, train_y)):
    mtcnn_image_filename = old_image_path.rsplit('/')[-1].rsplit('.')[0] + '.png'
    source_file = os.path.join(args.raw_data_path, mtcnn_image_filename)
    target_file = os.path.join(args.train_path, str(class_id), mtcnn_image_filename)
    # copy data to specific class folder
    
    if os.path.isfile(source_file):
      copyfile(source_file, target_file)
    else:
      num_ignore_train_y += 1
      writer.writerow([source_file])
    
print('Total images: {}'.format(num_origin_train_y))
print('Total failed images: {}'.format(num_ignore_train_y))
print('Total successful images: {}'.format(num_origin_train_y - num_ignore_train_y))

