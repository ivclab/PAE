python src/train.py \
--model_dir official_checkpoint/facenet \
--change_weight_name_from_github official_checkpoint/github/model-20180402-114759.ckpt-275
--task_id 1
--open_ratio 1.0 \
...

python src/train.py \
--model_dir official_checkpoint/facenet \
--max_epochs 4
--use_pruning_strategy \
--begin_pruning_epoch 0 \
--end_pruning_epoch 1.0 \
--pruning_hparams name=pruning,initial_sparsity=0.0,target_sparsity=0.1,pruning_frequency=10 \
...

python src/evaluate.py \
--model official_checkpoint/facenet/model.ckpt-1 \
--task_id 1 \
--print_mem \
--print_mask_info \
--verbose \
...


python src/train.py \
--logs_base_dir logs/age \
--models_base_dir official_checkpoint/age \
--data_dir /home/ivclab/fevemania/datasets/age_gender \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate 0.01 \
--max_nrof_epochs 150 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--validate_every_n_epochs 1 \
--gpu_memory_fraction 0.8 \
--pretrained_model official_checkpoint/facenet/model-.ckpt-8 \
--task_name age \
--task_id 2 \
--set_zeros_in_masks_to_current_task_id \
--open_ratio 1.0 \


end_epoch_no = 0
num_epoch_for_retrain_per_prune = 4
start_sparsity_list = (0.0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8)
end_sparsity_list   = (0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9)


for index = 0 to 8

  end_epoch_no = end_epoch_no + num_epoch_for_retrain_per_prune

  CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --models_dir official_checkpoint/facenet \
  --task_id 1 \
  --max_epochs ${end_epoch_no} \
  --use_pruning_strategy \
  --begin_pruning_epoch 0 \
  --end_pruning_epoch 1.0 \
  --pruning_hparams name=pruning,initial_sparsity=${start_sparsity_list[index]},target_sparsity=${end_sparsity_list[index]},pruning_frequency=10
  ...

  CUDA_VISIBLE_DEVICES=1 python src/evaluate.py \
  --model official_checkpoint/facenet/model.ckpt-$end_epoch_no \
  --task_id 1 \
  --print_mem \
  --print_mask_info \
  ...


python src/train.py \
--logs_base_dir/age \
--models_dir official_checkpoint/age \
--pretrained_model official_checkpoint/facenet/model-.ckpt-8 \
--task_id 2 \
--open_ratio 1.0 \
...

python src/evaluate.py \
--model official_checkpoint/age/model-.ckpt-85 \
--task_id 2 \
--print_mem \
--print_mask_info \
--verbose \
...