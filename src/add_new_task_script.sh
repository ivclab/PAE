# bash ./src/add_new_task_script.sh 0 chalearn/gender 2 experiment2/chalearn/gender/3 official_checkpoint/facenet/0/model-.ckpt-8
# bash ./src/add_new_task_script.sh 0 emotion 2 experiment2/chalearn/gender/5 official_checkpoint/experiment2/emotion/1/model-.ckpt-71
# bash ./src/add_new_task_script.sh 0 chalearn/gender 3 experiment2/chalearn/gender/5 official_checkpoint/experiment2/emotion/1/model-.ckpt-71
# bash ./src/add_new_task_script.sh 0 chalearn/gender 3 experiment2/chalearn/gender/expand official_checkpoint/experiment2/chalearn/gender/5/model-.ckpt-5
# bash ./src/add_new_task_script.sh 0 chalearn/gender 3 experiment2/chalearn/gender/expand experiment2/chalearn/gender/expand/model.ckpt-5-adjust
# bash ./src/add_new_task_script.sh 0 emotion 2 experiment2/chalearn/gender/expand experiment2/chalearn/gender/expand/model-.ckpt-5-adjust
# bash ./src/add_new_task_script.sh 0 chalearn/gender 3 experiment2/chalearn/gender/expand official_checkpoint/experiment2/chalearn/gender/0/model-.ckpt-41
# bash ./src/add_new_task_script.sh 2 chalearn/gender 3 experiment2/chalearn/gender/expand_final official_checkpoint/experiment2/emotion/weighted_loss/model-.ckpt-52

NUM_RUNS=9
nrof_epoch_to_run=5
start_sparsity_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
end_sparsity_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

GPU_ID=$1
TASK_NAME=$2
TASK_ID=$3
MODEL_FOLDER_NAME=$4
pretrained_model=$5


CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
--logs_base_dir 'logs/'$MODEL_FOLDER_NAME \
--models_base_dir 'pae_checkpoint/'$MODEL_FOLDER_NAME \
--data_dir 'data/'$TASK_NAME \
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
--pretrained_model $pretrained_model \
--task_name $TASK_NAME \
--task_id $TASK_ID \
--special_operation expand \
--filters_expand_ratio 1.2 \
--history_filters_expand_ratios 1.0,1.2 \
--share_only_task_1 \
--open_ratio 1.2 \
--learning_rate 0.0005 \

CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
--logs_base_dir 'logs/'$MODEL_FOLDER_NAME \
--models_base_dir 'pae_checkpoint/'$MODEL_FOLDER_NAME \
--data_dir 'data/'$TASK_NAME \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate 0.03 \
--nrof_addiitonal_epochs_to_run 500 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--validate_every_n_epochs 1 \
--gpu_memory_fraction 0.8 \
--task_name $TASK_NAME \
--task_id $TASK_ID \
--max_to_keep 1 \
--share_only_task_1 \
--filters_expand_ratio 1.2 \
--history_filters_expand_ratios 1.0,1.2 \

# Before pruning, write unpruning status into csv file
CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
--data_dir 'data/'$TASK_NAME \
--model 'pae_checkpoint/'$MODEL_FOLDER_NAME \
--use_fixed_image_standardization \
--task_name $TASK_NAME \
--task_id $TASK_ID \
--eval_once \
--print_mem \
--print_mask_info \
--share_only_task_1 \
--filters_expand_ratio 1.2 \
--history_filters_expand_ratios 1.0,1.2 \
--csv_file_path 'csv/'$MODEL_FOLDER_NAME'.csv' \

for RUN_ID in `seq 0 $((NUM_RUNS-1))`; do
   
  CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
  --logs_base_dir 'logs/'$MODEL_FOLDER_NAME \
  --models_base_dir 'pae_checkpoint/'$MODEL_FOLDER_NAME \
  --data_dir 'data/'$TASK_NAME \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --optimizer ADAM \
  --learning_rate 0.0002 \
  --nrof_addiitonal_epochs_to_run $nrof_epoch_to_run \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --use_fixed_image_standardization \
  --weight_decay 5e-4 \
  --embedding_size 512 \
  --validate_every_n_epochs 1 \
  --gpu_memory_fraction 0.8 \
  --task_name $TASK_NAME \
  --task_id $TASK_ID \
  --max_to_keep $nrof_epoch_to_run \
  --use_pruning_strategy \
  --begin_pruning_epoch 0 \
  --end_pruning_epoch 1.0 \
  --pruning_hparams name=pruning,initial_sparsity=${start_sparsity_list[RUN_ID]},target_sparsity=${end_sparsity_list[RUN_ID]},pruning_frequency=10 \
  --share_only_task_1 \
  --filters_expand_ratio 1.2 \
  --history_filters_expand_ratios 1.0,1.2

  CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
  --data_dir 'data/'$TASK_NAME \
  --model 'pae_checkpoint/'$MODEL_FOLDER_NAME \
  --use_fixed_image_standardization \
  --task_name $TASK_NAME \
  --task_id $TASK_ID \
  --eval_once \
  --print_mem \
  --print_mask_info \
  --csv_file_path 'csv/'$MODEL_FOLDER_NAME'.csv' \
  --share_only_task_1 \
  --filters_expand_ratio 1.2 \
  --history_filters_expand_ratios 1.0,1.2
done
