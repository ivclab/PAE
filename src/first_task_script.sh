#!/bin/bash

# bash src/first_task_script.sh

NUM_RUNS=9
nrof_epoch_to_run=4
start_sparsity_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
end_sparsity_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

GPU_ID=0
TASK_NAME="facenet"
TASK_ID=1
MODEL_FOLDER_NAME="facenet/0"

CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
--models_base_dir 'pae_checkpoint/'$MODEL_FOLDER_NAME \
--logs_base_dir 'logs/'$MODEL_FOLDER_NAME \
--lfw_dir 'data/lfw_mtcnnpy_160' \
--data_dir 'data/vggface2_mtcnn_182' \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--validate_every_n_epochs 1 \
--gpu_memory_fraction 0.8 \
--change_weight_name_from_github official_checkpoint/model-20180402-114759.ckpt-275 \
--task_name $TASK_NAME \
--task_id $TASK_ID \
--open_ratio 1.0 \

for RUN_ID in `seq 0 $((NUM_RUNS-1))`; do
   
  CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
  --models_base_dir 'pae_checkpoint/'$MODEL_FOLDER_NAME \
  --logs_base_dir 'logs/'$MODEL_FOLDER_NAME \
  --lfw_dir 'data/lfw_mtcnnpy_160' \
  --data_dir 'data/vggface2_mtcnn_182' \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --optimizer ADAM \
  --learning_rate 0.0005 \
  --nrof_addiitonal_epochs_to_run $nrof_epoch_to_run \
  --keep_probability 0.8 \
  --random_flip \
  --random_crop \
  --use_fixed_image_standardization \
  --weight_decay 5e-4 \
  --embedding_size 512 \
  --prelogits_norm_loss_factor 5e-4 \
  --lfw_distance_metric 1 \
  --lfw_use_flipped_images \
  --lfw_subtract_mean \
  --validation_set_split_ratio 0.05 \
  --validate_every_n_epochs 1 \
  --gpu_memory_fraction 0.8 \
  --task_name $TASK_NAME \
  --task_id $TASK_ID \
  --max_to_keep $nrof_epoch_to_run \
  --use_pruning_strategy \
  --begin_pruning_epoch 0 \
  --end_pruning_epoch 1.0 \
  --pruning_hparams name=pruning,initial_sparsity=${start_sparsity_list[RUN_ID]},target_sparsity=${end_sparsity_list[RUN_ID]},pruning_frequency=10 \

  CUDA_VISIBLE_DEVICES=$GPU_ID python src/validate_on_lfw.py \
  --lfw_dir 'data/lfw_mtcnnpy_160' \
  --lfw_pairs 'data/pairs.txt' \
  --model 'pae_checkpoint/'$MODEL_FOLDER_NAME \
  --use_fixed_image_standardization \
  --task_name $TASK_NAME \
  --task_id $TASK_ID \
  --distance_metric 1 \
  --use_flipped_images \
  --subtract_mean \
  --eval_once \
  --print_mem \
  --print_mask_info \
  --csv_file_path 'csv/'$MODEL_FOLDER_NAME'.csv'

done
