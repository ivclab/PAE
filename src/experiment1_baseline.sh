GPU_ID=3
dataset=("age/test_fold_is_0" "gender/test_fold_is_0" "age/test_fold_is_1" "gender/test_fold_is_1" "age/test_fold_is_2" "gender/test_fold_is_2" "age/test_fold_is_3" "gender/test_fold_is_3" "age/test_fold_is_4" "gender/test_fold_is_4")
MODEL_FOLDER_NAME=("age/test_fold_is_0" "gender/test_fold_is_0" "age/test_fold_is_1" "gender/test_fold_is_1" "age/test_fold_is_2" "gender/test_fold_is_2" "age/test_fold_is_3" "gender/test_fold_is_3" "age/test_fold_is_4" "gender/test_fold_is_4")

NUM_RUNS=10

for idx in `seq 0 $((NUM_RUNS-1))`
do
  CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
  --logs_base_dir 'logs/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune' \
  --models_base_dir 'official_checkpoint/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune' \
  --data_dir '/home/ivclab/fevemania/datasets/'${dataset[idx]} \
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
  --change_weight_name_from_github official_checkpoint/model-20180402-114759.ckpt-275 \
  --open_ratio 1.0 \
  --task_name ${dataset[idx]} \
  --checkpoint_exclude_scopes=task_1/Logits \
  --task_id 1

  CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
  --logs_base_dir 'logs/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune' \
  --models_base_dir 'official_checkpoint/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune' \
  --data_dir '/home/ivclab/fevemania/datasets/'${dataset[idx]} \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --optimizer ADAM \
  --learning_rate 1e-2 \
  --nrof_addiitonal_epochs_to_run 3 \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --use_fixed_image_standardization \
  --weight_decay 5e-4 \
  --embedding_size 512 \
  --validate_every_n_epochs 1 \
  --gpu_memory_fraction 0.8 \
  --task_name ${dataset[idx]} \
  --task_id 1 \
  --max_to_keep 1 \
  --trainable_scopes task_1/Logits

  CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_softmax.py \
  --logs_base_dir 'logs/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune' \
  --models_base_dir 'official_checkpoint/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune' \
  --data_dir '/home/ivclab/fevemania/datasets/'${dataset[idx]} \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --optimizer ADAM \
  --learning_rate 1e-2 \
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
  --task_id 1 \
  --max_to_keep 1

  CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
  --data_dir '/home/ivclab/fevemania/datasets/'${dataset[idx]} \
  --model 'official_checkpoint/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune' \
  --use_fixed_image_standardization \
  --task_id 1 \
  --task_name $dataset \
  --print_mem \
  --print_mask_info \
  --eval_once \
  --csv_file_path 'csv/experiement1/'${MODEL_FOLDER_NAME[idx]}'_finetune.csv' 

done