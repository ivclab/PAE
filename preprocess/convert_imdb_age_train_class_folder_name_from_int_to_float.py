import os
import sys
import argparse
import pdb
from pprint import pprint

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--data_path', type=str, default='/home/ivclab/fevemania/datasets/imdb/age/train',
    help='folder that contains training images')
args = FLAGS.parse_args()


if os.path.exists(args.data_path):
  class_folders = [item for item in os.listdir(args.data_path) if item.isdigit()]
  for old_filename in class_folders:
    new_filename = str(float(old_filename))
    os.rename(os.path.join(args.data_path, old_filename), 
              os.path.join(args.data_path, new_filename))
else:
  print("{} doesn't exist".format(args.data_path))