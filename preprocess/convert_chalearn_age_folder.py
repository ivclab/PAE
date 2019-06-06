import os
import pandas as pd
from pprint import pprint
from shutil import copyfile
PATH='/home/ivclab/datasets/chalearn/age/val'

NEW_PATH='/home/ivclab/datasets/chalearn/age/float'
csv_path='/home/ivclab/fevemania/facenet/preprocess/valid_gt.csv'

old_file_paths = {f: os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.png'}

records = pd.read_csv(csv_path)
for _, record in records.iterrows():
  png_image = record['image'].replace('jpg', 'png')
  store_folder = os.path.join(NEW_PATH, str(record['mean']))
  store_path = os.path.join(store_folder, png_image)
  try:
    old_path = old_file_paths[png_image]
    if not os.path.exists(store_folder):
      os.makedirs(store_folder)
    copyfile(old_path, store_path)
  except:
    pass


# for item in os.walk(PATH):
#   filenames = item[2]
#   for f in filenames:
#     if os.path.splitext(f)[1] == '.png':
#       print(item[0])
  # break

