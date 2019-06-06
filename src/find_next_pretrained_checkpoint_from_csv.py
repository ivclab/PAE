import csv
import argparse
import pandas as pd
import pdb

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--csv_path', type=str, default='/home/ivclab/fevemania/prac_DL/facenet/csv/test.csv',
    help="csv file to store the history of last task's pruning ratio and its corresponding accuracy")
args = FLAGS.parse_args()

history = pd.read_csv(args.csv_path)
pdb.set_trace()
history['Acc']


print('here')
# val_x = val_data_info[''].tolist()
# val_y = val_data_info['expression'].tolist()
# num_origin_val_y = len(val_y)
# classes = list(set(val_y))
# num_classes = len(classes)
