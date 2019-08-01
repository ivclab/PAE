# Usage:
# bash ./src/finetune_script.sh 1 emotion experiment2/emotion
# bash ./src/finetune_script.sh 3 chalearn/gender experiment2/chalearn/gender
# bash ./src/finetune_script.sh 3 chalearn/age experiment2/chalearn/age_1
# bash ./src/finetune_script.sh 0 an_chalearn/age experiment2/an_chalearn/age
# bash ./src/finetune_script.sh 0 emotion1 experiment2/emotion/weighted_loss

GPU_ID=$1
dataset=$2
MODEL_FOLDER_NAME=$3

# CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
# --logs_base_dir 'logs/'$MODEL_FOLDER_NAME'_finetune' \
# --models_base_dir 'official_checkpoint/'$MODEL_FOLDER_NAME'_finetune' \
# --data_dir '/home/ivclab/fevemania/datasets/'$dataset \
# --image_size 160 \
# --model_def models.inception_resnet_v1 \
# --optimizer ADAM \
# --learning_rate 1e-2 \
# --keep_probability 0.8 \
# --random_crop \
# --random_flip \
# --use_fixed_image_standardization \
# --weight_decay 5e-4 \
# --embedding_size 512 \
# --validate_every_n_epochs 1 \
# --gpu_memory_fraction 0.8 \
# --change_weight_name_from_github official_checkpoint/model-20180402-114759.ckpt-275 \
# --open_ratio 1.0 \
# --task_name $dataset \
# --checkpoint_exclude_scopes=task_1/Logits \
# --task_id 1 \

# CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
# --logs_base_dir 'logs/'$MODEL_FOLDER_NAME'_finetune' \
# --models_base_dir 'official_checkpoint/'$MODEL_FOLDER_NAME'_finetune' \
# --data_dir '/home/ivclab/fevemania/datasets/'$dataset \
# --image_size 160 \
# --model_def models.inception_resnet_v1 \
# --optimizer ADAM \
# --learning_rate 1e-2 \
# --nrof_addiitonal_epochs_to_run 3 \
# --keep_probability 0.8 \
# --random_crop \
# --random_flip \
# --use_fixed_image_standardization \
# --weight_decay 5e-4 \
# --embedding_size 512 \
# --validate_every_n_epochs 1 \
# --gpu_memory_fraction 0.8 \
# --task_name $dataset \
# --task_id 1 \
# --max_to_keep 1 \
# --trainable_scopes task_1/Logits \

# CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
# --logs_base_dir 'logs/'$MODEL_FOLDER_NAME'_finetune' \
# --models_base_dir 'official_checkpoint/'$MODEL_FOLDER_NAME'_finetune' \
# --data_dir '/home/ivclab/fevemania/datasets/'$dataset \
# --image_size 160 \
# --model_def models.inception_resnet_v1 \
# --optimizer ADAM \
# --learning_rate 2e-3 \
# --nrof_addiitonal_epochs_to_run 500 \
# --keep_probability 0.8 \
# --random_crop \
# --random_flip \
# --use_fixed_image_standardization \
# --weight_decay 5e-4 \
# --embedding_size 512 \
# --validate_every_n_epochs 1 \
# --gpu_memory_fraction 0.8 \
# --task_name $dataset \
# --task_id 1 \
# --max_to_keep 1 \

CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
--model 'official_checkpoint/'$MODEL_FOLDER_NAME'_finetune' \
--data_dir '/home/ivclab/fevemania/datasets/'$dataset \
--use_fixed_image_standardization \
--task_id 1 \
--task_name $dataset \
--print_mem \
--print_mask_info \
--eval_once \
--csv_file_path 'csv/'$MODEL_FOLDER_NAME'_finetune.csv' \