# Baseline: (finetune)
# model size (age):
# model size (gender): 
# Avg age acc:
# Avg gender acc:

# ------
# Pruned: (Task1: facenet, Task2: age, Task3: gender)
# Avg model size:
# Task1 specific part:
# Avg Task2 specific part:
# Avg Task3 specific part:  
# Task1 acc:
# Avg Task2 acc:
# Avg Task3 acc:


import csv
import argparse
import pandas as pd
import sys
import pdb

facenet_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/facenet/0.csv'
pruned_age_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment1/age/test_fold_is_{}.csv'
pruned_gender_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment1/gender/test_fold_is_{}.csv'
finetune_age_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment1/age/test_fold_is_{}_finetune.csv'
finetune_gender_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment1/gender/test_fold_is_{}_finetune.csv'

sum_pruned_age_acc = 0.0
sum_pruned_age_specific_model_size = 0.0
sum_pruned_gender_acc = 0.0
sum_pruned_gender_specific_model_size = 0.0
sum_finetune_age_acc = 0.0
sum_finetune_gender_acc = 0.0

sum_total_pruned_model_size = 0.0
shared_part_for_all_task = 0.0

age_finetune_model_size = 0.0
gender_finetune_model_size = 0.0


df = pd.read_csv(facenet_csv)
pretrained_facenet_acc_mean = df.loc[0]['Acc Mean']*100
pretrained_facenet_acc_std = df.loc[0]['Acc Std']*100
pretrained_facenet_size = df.loc[0]['Model size through inference (MB) (Shared part + task-specific part)'] - df.loc[0]['Whole masks (MB)']
pruned_facenet_model_size = df.loc[df['Acc Mean'].idxmax()]['Task specific part (MB)']
pruned_facenet_acc_mean = df['Acc Mean'].max()*100
pruned_facenet_acc_std = df['Acc Std'].max()*100

for i in range(5):
  df = pd.read_csv(pruned_age_csv.format(i))
  sum_pruned_age_acc += df['Acc'].max()
  sum_pruned_age_specific_model_size += df.loc[df['Acc'].idxmax()]['Task specific part (MB)']
  sum_total_pruned_model_size += df.loc[df['Acc'].idxmax()]['Task specific part (MB)']
  if i == 0:
    shared_part_for_all_task = df['Shared part (MB)'].max()

  df = pd.read_csv(pruned_gender_csv.format(i))
  sum_pruned_gender_acc += df['Acc'].max()
  sum_pruned_gender_specific_model_size += df.loc[df['Acc'].idxmax()]['Task specific part (MB)']
  sum_total_pruned_model_size += df.loc[df['Acc'].idxmax()]['Task specific part (MB)']
  # print("pruned: {}".format(df['Acc'].max()))

  df = pd.read_csv(finetune_age_csv.format(i))
  sum_finetune_age_acc += df['Acc'].max()
  if i == 0:
    age_finetune_model_size = df['Model size through inference (MB) (Shared part + task-specific part)'].max() - df['Whole masks (MB)'].max()

  df = pd.read_csv(finetune_gender_csv.format(i))
  sum_finetune_gender_acc += df['Acc'].max()
  if i == 0:
    gender_finetune_model_size = df['Model size through inference (MB) (Shared part + task-specific part)'].max() - df['Whole masks (MB)'].max()

  # print("fine {}".format(df['Acc'].max()))

avg_pruned_age_acc = sum_pruned_age_acc/5*100
avg_pruned_gender_acc = sum_pruned_gender_acc/5*100
avg_finetune_age_acc = sum_finetune_age_acc/5*100
avg_finetune_gender_acc = sum_finetune_gender_acc/5*100

avg_total_pruned_model_size = sum_total_pruned_model_size/5
avg_total_pruned_model_size += pruned_facenet_model_size # This already contains `shared_part_for_all_task`

avg_pruned_age_specific_model_size = sum_pruned_age_specific_model_size/5
avg_pruned_gender_specific_model_size = sum_pruned_gender_specific_model_size/5


print('Baseline: (finetune)')
print('--------------------')
print('Model size (facenet): {:.3f} MB'.format(pretrained_facenet_size))
print('Model size (age): {:.3f} MB'.format(age_finetune_model_size))
print('Model size (gender): {:.3f} MB'.format(gender_finetune_model_size))
print('Pretrained accuracy (facenet): {:.3f} +- {:.3f} %'.format(pretrained_facenet_acc_mean, pretrained_facenet_acc_std))
print('Average accuracy (age): {:.3f} %'.format(avg_finetune_age_acc))
print('Average accuracy (gender): {:.3f} %'.format(avg_finetune_gender_acc))
print()
print('Pruned: (Task1: facenet, Task2: age, Task3: gender)')
print('--------------------------------------------------')
print('Average total model size: {:.3f} MB'.format(avg_total_pruned_model_size))
print('shared part (Task 1 feature) for all task: {:.3f} MB'.format(shared_part_for_all_task))
print('Model size (facenet): {:.3f} MB'.format(pruned_facenet_model_size))
print('Average model size (age): {:.3f} MB'.format(avg_pruned_age_specific_model_size))
print('Average model size (gender): {:.3f} MB'.format(avg_pruned_gender_specific_model_size))
print('Accuracy (facenet): {:.3f} +- {:.3f} %'.format(pruned_facenet_acc_mean, pruned_facenet_acc_std))
print('Average accuracy (age): {:.3f} %'.format(avg_pruned_age_acc))
print('Average accuracy (gender): {:.3f} %'.format(avg_pruned_gender_acc))
