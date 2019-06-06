import os
import sys
import argparse
import pdb
from pprint import pprint
import re
from distutils.dir_util import copy_tree

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--chalearn_data_path', type=str, default='/home/ivclab/fevemania/datasets/chalearn/age/train',
    help='')

FLAGS.add_argument('--imdb_data_path', type=str, default='/home/ivclab/fevemania/datasets/imdb/age/train',
    help='')

args = FLAGS.parse_args()


if os.path.exists(args.chalearn_data_path) and os.path.exists(args.imdb_data_path):
  # class_folders_in_chalearn = [item for item in os.listdir(args.chalearn_data_path) if item.isdigit()]
  class_folders_in_chalearn = os.listdir(args.chalearn_data_path)
  float_pattern="\d+\.\d+"
  
  # pprint()
  for folder_name in class_folders_in_chalearn:
    _ = re.match(float_pattern, folder_name)
    if _:
      copy_tree(os.path.join(args.chalearn_data_path, folder_name), 
                os.path.join(args.imdb_data_path, folder_name))
    else:
      pass
else:
  print("{} or {} doesn't exist".format(
      args.chalearn_data_path, args.imdb_data_path))