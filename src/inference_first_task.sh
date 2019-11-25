#!/bin/bash

# bash src/inference_first_task.sh

GPU_ID=0
MODEL_FOLDER_NAME="facenet/0"

CUDA_VISIBLE_DEVICES=$GPU_ID python src/validate_on_lfw.py \
--lfw_dir '~/fevemania/datasets/lfw_mtcnnpy_160' \
--lfw_pairs 'data/pairs.txt' \
--model 'official_checkpoint/'$MODEL_FOLDER_NAME \
--use_fixed_image_standardization \
--task_name 'facenet' \
--task_id 1 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--eval_once \
--print_mem \
--print_mask_info 