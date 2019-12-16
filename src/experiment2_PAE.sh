#!/bin/bash
# Usage:

nrof_epoch_to_run=5
start_sparsity_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
end_sparsity_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

GPU_ID=0
dataset=("emotion" "chalearn/gender")
PREVIOUS_MODEL_FOLDER_NAME=("facenet/0" "experiment2/emotion/weighted_loss")
MODEL_FOLDER_NAME=("experiment2/emotion/weighted_loss" "experiment2/chalearn/gender")
TASK_ID=(2 3)

for idx in `seq 0 1`
do
  python src/find_prev_task_best_checkpoint_as_pretrained_model_for_curr_task.py \
  --prev_task_folder_name ${PREVIOUS_MODEL_FOLDER_NAME[idx]} \
  --curr_task_folder_name ${MODEL_FOLDER_NAME[idx]} \
  --curr_task_id ${TASK_ID[idx]} \

  CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
  --logs_base_dir 'logs/'${MODEL_FOLDER_NAME[idx]} \
  --models_base_dir 'pae_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
  --data_dir 'data/'${dataset[idx]} \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --optimizer ADAM \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --use_fixed_image_standardization \
  --weight_decay 5e-4 \
  --embedding_size 512 \
  --validate_every_n_epochs 1 \
  --gpu_memory_fraction 0.8 \
  --has_pretrained_model_for_curr_task \
  --task_name ${dataset[idx]} \
  --task_id ${TASK_ID[idx]} \
  --open_ratio 1.0 \

  # Finetune new task
  CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
  --logs_base_dir 'logs/'${MODEL_FOLDER_NAME[idx]} \
  --models_base_dir 'pae_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
  --data_dir 'data/'${dataset[idx]} \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --optimizer ADAM \
  --learning_rate 0.01 \
  --nrof_addiitonal_epochs_to_run 500 \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --use_fixed_image_standardization \
  --weight_decay 5e-4 \
  --embedding_size 512 \
  --validate_every_n_epochs 1 \
  --gpu_memory_fraction 0.8 \
  --task_name ${dataset[idx]} \
  --task_id ${TASK_ID[idx]} \
  --max_to_keep 1    

  CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
  --data_dir 'data/'${dataset[idx]} \
  --model 'pae_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
  --use_fixed_image_standardization \
  --task_name ${dataset[idx]} \
  --task_id ${TASK_ID[idx]} \
  --eval_once \
  --print_mem \
  --print_mask_info \
  --csv_file_path 'csv/'${MODEL_FOLDER_NAME[idx]}'.csv' \

  for RUN_ID in `seq 0 8`; do
     
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
    --logs_base_dir 'logs/'${MODEL_FOLDER_NAME[idx]} \
    --models_base_dir 'pae_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
    --data_dir 'data/'${dataset[idx]} \
    --image_size 160 \
    --model_def models.inception_resnet_v1 \
    --optimizer ADAM \
    --learning_rate 0.0005 \
    --nrof_addiitonal_epochs_to_run $nrof_epoch_to_run \
    --keep_probability 0.8 \
    --random_crop \
    --random_flip \
    --use_fixed_image_standardization \
    --weight_decay 5e-4 \
    --embedding_size 512 \
    --validate_every_n_epochs 1 \
    --gpu_memory_fraction 0.8 \
    --task_name ${dataset[idx]} \
    --task_id ${TASK_ID[idx]} \
    --max_to_keep $nrof_epoch_to_run \
    --use_pruning_strategy \
    --begin_pruning_epoch 0 \
    --end_pruning_epoch 1.0 \
    --pruning_hparams name=pruning,initial_sparsity=${start_sparsity_list[RUN_ID]},target_sparsity=${end_sparsity_list[RUN_ID]},pruning_frequency=10

    CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
    --data_dir 'data/'${dataset[idx]} \
    --model 'pae_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
    --use_fixed_image_standardization \
    --task_name ${dataset[idx]} \
    --task_id ${TASK_ID[idx]} \
    --eval_once \
    --print_mem \
    --print_mask_info \
    --csv_file_path 'csv/'${MODEL_FOLDER_NAME[idx]}'.csv' \

  done
done