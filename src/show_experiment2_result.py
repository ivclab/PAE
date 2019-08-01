import csv
import argparse
import pandas as pd
import sys
import pdb

facenet_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/facenet/0.csv'

finetune_emotion_without_weighted_loss_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment2/emotion/finetune_without_weighted_loss.csv'
finetune_emotion_with_weighted_loss_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment2/emotion/finetune_with_weighted_loss.csv'
finetune_gender_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment2/chalearn/gender_finetune.csv'


pruned_emotion_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment2/emotion/weighted_loss.csv' 
pruned_gender_csv = '/home/ivclab/fevemania/prac_DL/facenet/csv/experiment2/chalearn/gender/expand_final_05.csv'

total_pruned_model_size = 0.0

df = pd.read_csv(facenet_csv)
pretrained_facenet_acc_mean = df.loc[0]['Acc Mean']*100
pretrained_facenet_acc_std = df.loc[0]['Acc Std']*100
pretrained_facenet_size = df.loc[0]['Model size through inference (MB) (Shared part + task-specific part)'] - df.loc[0]['Whole masks (MB)']
pruned_facenet_model_size = df.loc[df['Acc Mean'].idxmax()]['Task specific part (MB)']
pruned_facenet_acc_mean = df['Acc Mean'].max()*100
pruned_facenet_acc_std = df['Acc Std'].max()*100
total_pruned_model_size += pruned_facenet_model_size


# emotion(finetune)
df = pd.read_csv(finetune_emotion_without_weighted_loss_csv)
finetune_emotion_wo_acc = df['Acc'].max()*100
finetune_emotion_wo_model_size = df['Model size through inference (MB) (Shared part + task-specific part)'].max() - df['Whole masks (MB)'].max()
df = pd.read_csv(finetune_emotion_with_weighted_loss_csv)
finetune_emotion_w_acc = df['Acc'].max()*100
finetune_emotion_w_model_size = df['Model size through inference (MB) (Shared part + task-specific part)'].max() - df['Whole masks (MB)'].max()

# chalearn/gender (finetune)
df = pd.read_csv(finetune_gender_csv)
finetune_gender_acc = df['Acc'].max()*100
finetune_gender_model_size = df['Model size through inference (MB) (Shared part + task-specific part)'].max() - df['Whole masks (MB)'].max()


# emotion(pruned)
df = pd.read_csv(pruned_emotion_csv)
pruned_emotion_acc = df['Acc'].max()*100
pruned_emotion_specific_model_size = df.loc[df['Acc'].idxmax()]['Task specific part (MB)']
total_pruned_model_size += df.loc[df['Acc'].idxmax()]['Task specific part (MB)']
shared_part_for_all_task = df['Shared part (MB)'].max()

df = pd.read_csv(pruned_gender_csv)
pruned_gender_acc = df['Acc'][7:].max()*100
pruned_gender_speicific_model_size = df.loc[df['Acc'][7:].idxmax()]['Task specific part (MB)']
total_pruned_model_size += df.loc[df['Acc'][7:].idxmax()]['Task specific part (MB)']

print('Baseline: (finetune)')
print('--------------------')
print('Model size (facenet): {:.3f} MB'.format(pretrained_facenet_size))
print('Model size (emotion) (w/o weighted_loss): {:.3f} MB'.format(finetune_emotion_wo_model_size))
print('Model size (emotion) (with weighted_loss): {:.3f} MB'.format(finetune_emotion_w_model_size))
print('Model size (chalearn/gender): {:.3f} MB'.format(finetune_gender_model_size))
print('Pretrained accuracy (facenet): {:.3f} +- {:.3f} %'.format(pretrained_facenet_acc_mean, pretrained_facenet_acc_std))
print('Accuracy (emotion) (w/o weighted_loss): {:.3f} %'.format(finetune_emotion_wo_acc))
print('Accuracy (emotion) (with weighted_loss): {:.3f} %'.format(finetune_emotion_w_acc))
print('Accuracy (chalearn/gender): {:.3f} %'.format(finetune_gender_acc))
print()
print('Pruned: (Task1: facenet, Task2: emotion, Task3: chalearn/gender)')
print('--------------------------------------------------')
print('Total model size: {:.3f} MB'.format(total_pruned_model_size))
print('shared part (Task 1 feature) for all task: {:.3f} MB'.format(shared_part_for_all_task))
print('Model size (facenet): {:.3f} MB'.format(pruned_facenet_model_size))
print('Model size (emotion) (with weighted_loss): {:.3f} MB'.format(pruned_emotion_specific_model_size))
print('Model size (chalearn/gender): {:.3f} MB'.format(pruned_gender_speicific_model_size))
print('Accuracy (facenet): {:.3f} +- {:.3f} %'.format(pruned_facenet_acc_mean, pruned_facenet_acc_std))
print('Accuracy (emotion): {:.3f} %'.format(pruned_emotion_acc))
print('Accuracy (chalearn/gender): {:.3f} %'.format(pruned_gender_acc))

