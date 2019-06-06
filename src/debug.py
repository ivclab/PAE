# Preprocessing dataset
CUDA_VISIBLE_DEVICES=3 python src/align/align_dataset_mtcnn.py \
~/fevemania/datasets/age_gender/others/test_fold_is_4/train/age \
~/fevemania/datasets/age/test_fold_is_4/train \
--image_size 182 \
--margin 44

CUDA_VISIBLE_DEVICES=3 python src/align/align_dataset_mtcnn.py \
~/fevemania/datasets/age_gender/others/test_fold_is_4/train/gender \
~/fevemania/datasets/gender/test_fold_is_4/train \
--image_size 182 \
--margin 44

CUDA_VISIBLE_DEVICES=2 python src/align/align_dataset_mtcnn.py \
~/fevemania/datasets/age_gender/others/test_fold_is_4/validation/age \
~/fevemania/datasets/age/test_fold_is_4/val \
--image_size 182 \
--margin 44

CUDA_VISIBLE_DEVICES=3 python src/align/align_dataset_mtcnn.py \
~/fevemania/datasets/age_gender/others/test_fold_is_4/validation/gender \
~/fevemania/datasets/gender/test_fold_is_4/val \
--image_size 182 \
--margin 44


CUDA_VISIBLE_DEVICES=3 python src/validation.py \
--data_dir '~/fevemania/datasets/facenet' \
--model 'official_checkpoint/emotion/1/copied' \
--use_fixed_image_standardization \
--task_name emotion \
--task_id 2 \
--eval_once \
--print_mem \
--print_mask_info \


# step1 : Transfer official checkpoint into the one that has our self-defined variables 
# ie. masks and their thresholds, and change weights' names in checkpoint file (from '' to 'task_1')
#
# In the meanwhile, add hole for task_1
CUDA_VISIBLE_DEVICES=0 python src/train_softmax.py \
--logs_base_dir logs/facenet/5 \
--models_base_dir official_checkpoint/facenet/5 \
--data_dir /home/ivclab/fevemania/datasets/vggface2 \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir ~/datasets/lfw_mtcnnpy_160/ \
--optimizer ADAM \
--learning_rate 0.0005 \
--nrof_addiitonal_epochs_to_run 150 \
--keep_probability 0.4 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4 \
--gpu_memory_fraction 0.8 \
--change_weight_name_from_github official_checkpoint/model-20180402-114759.ckpt-275 \
--open_ratio 1.0 \
--task_id 1 \

# step2 : use csv file to record the initial accuracy and its pruned status (pruned_ratio = 0.0)
CUDA_VISIBLE_DEVICES=1 python src/validate_on_lfw.py \
--lfw_dir ~/fevemania/datasets/lfw_mtcnnpy_160 \
--lfw_pairs data/pairs.txt \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization \
--task_id 1 \
--model official_checkpoint/facenet/5/model-.ckpt-0 \
--eval_once \
# --print_mem \
# --print_mask_info \
# --verbose \
# --csv_file_path csv/facenet.csv \

# step3 : gradually pruning (each time prung 10%)
CUDA_VISIBLE_DEVICES=0 python src/train_softmax.py \
--logs_base_dir logs/facenet \
--models_base_dir official_checkpoint/facenet \
--data_dir /home/ivclab/fevemania/datasets/vggface2/train_182 \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir ~/datasets/lfw_mtcnnpy_160/ \
--optimizer ADAM \
--learning_rate 0.0005 \
--nrof_addiitonal_epochs_to_run 4 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 1000 \
--prelogits_norm_loss_factor 5e-4 \
--gpu_memory_fraction 0.8 \
--use_pruning_strategy \
--begin_pruning_epoch 0 \
--end_pruning_epoch 1.0 \
--pruning_hparams name=pruning,initial_sparsity=0.0,target_sparsity=0.1,pruning_frequency=10 \

# Step 4. second task (age) -> open new hole
CUDA_VISIBLE_DEVICES=2 python src/train_softmax.py \
--logs_base_dir logs/age \
--models_base_dir official_checkpoint/age \
--data_dir /home/ivclab/fevemania/datasets/age_gender \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate 0.01 \
--nrof_addiitonal_epochs_to_run 150 \
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
--open_ratio 1.0 \

# Step 5. check if the past task is forgot by the network
CUDA_VISIBLE_DEVICES=3 python src/validate_on_lfw.py \
--lfw_dir ~/fevemania/datasets/lfw_mtcnnpy_160 \
--model official_checkpoint/age/model-.ckpt-52 \
--lfw_pairs data/pairs.txt \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization \
--task_id 1 \
--lfw_batch_size 300 \
--eval_once \
--print_mem \
--print_mask_info \

# Step 6. start training age (task2)
CUDA_VISIBLE_DEVICES=2 python src/train_softmax.py \
--logs_base_dir logs/age \
--models_base_dir official_checkpoint/age \
--data_dir /home/ivclab/fevemania/datasets/age_gender \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate 0.001 \
--nrof_addiitonal_epochs_to_run 60 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--validate_every_n_epochs 1 \
--gpu_memory_fraction 0.8 \
--task_name age \
--task_id 2 \
--max_to_keep 10

# Step 7. evaluate task2
CUDA_VISIBLE_DEVICES=1 python src/validation.py \
--data_dir ~/fevemania/datasets/age_gender \
--model official_checkpoint/age/model-.ckpt-85 \
--use_fixed_image_standardization \
--task_id 2 \
--print_mem \
--print_mask_info \
--verbose \
--csv_file_path csv/age.csv \
--eval_once \

# Step 8. pruning task2
CUDA_VISIBLE_DEVICES=0 python src/train_softmax.py \
--logs_base_dir logs/age \
--models_base_dir official_checkpoint/age \
--data_dir /home/ivclab/fevemania/datasets/age_gender \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate 0.0005 \
--nrof_addiitonal_epochs_to_run 85 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--validate_every_n_epochs 1 \
--gpu_memory_fraction 0.8 \
--task_name age \
--task_id 2 \
--max_to_keep 10 \
--use_pruning_strategy \
--begin_pruning_epoch 0 \
--end_pruning_epoch 1.0 \
--pruning_hparams name=pruning,initial_sparsity=0.8,target_sparsity=0.85,pruning_frequency=10 \



# Step 9. open hole for task3
CUDA_VISIBLE_DEVICES=0 python src/train_softmax.py \
--logs_base_dir logs/gender \
--models_base_dir official_checkpoint/gender \
--data_dir /home/ivclab/fevemania/datasets/age_gender \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate 0.01 \
--nrof_addiitonal_epochs_to_run 150 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--validate_every_n_epochs 1 \
--gpu_memory_fraction 0.8 \
--pretrained_model official_checkpoint/age/model-.ckpt-72 \
--task_name gender \
--task_id 3 \
--set_zeros_in_masks_to_current_task_id \
--open_ratio 1.0 \


# Step 10. check if the past task is forgot by the network
CUDA_VISIBLE_DEVICES=3 python src/validate_on_lfw.py \
--lfw_dir ~/fevemania/datasets/lfw_mtcnnpy_160 \
--model official_checkpoint/gender/model-.ckpt-0 \
--lfw_pairs data/pairs.txt \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization \
--task_id 1 \
--lfw_batch_size 300 \
--eval_once \
--print_mem \
--print_mask_info \


CUDA_VISIBLE_DEVICES=2 python src/validation.py \
--data_dir ~/fevemania/datasets/age_gender \
--model official_checkpoint/gender/model-.ckpt-15 \
--use_fixed_image_standardization \
--task_id 2 \
--task_name age \
--eval_once \
--print_mem \
--print_mask_info \


CUDA_VISIBLE_DEVICES=1 python src/validation.py \
--data_dir ~/fevemania/datasets/age_gender/others/test_fold_is_4 \
--model official_checkpoint/gender/copied/model-.ckpt-15 \
--use_fixed_image_standardization \
--task_id 2 \
--task_name age \
--eval_once \
--print_mem \
--print_mask_info \

# Step 11. start training gender (task3)
CUDA_VISIBLE_DEVICES=0 python src/train_softmax.py \
--logs_base_dir logs/gender \
--models_base_dir official_checkpoint/gender \
--data_dir /home/ivclab/fevemania/datasets/age_gender \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate 0.0005 \
--nrof_addiitonal_epochs_to_run 35 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--weight_decay 5e-4 \
--embedding_size 512 \
--validate_every_n_epochs 1 \
--gpu_memory_fraction 0.8 \
--task_name gender \
--task_id 3 \
--max_to_keep 5

# Stepp 12. evaluate task 3
CUDA_VISIBLE_DEVICES=2 python src/validation.py \
--data_dir ~/fevemania/datasets/age_gender \
--model official_checkpoint/gender/model-.ckpt-28 \
--use_fixed_image_standardization \
--task_id 3 \
--eval_once \
--task_name gender \
--print_mem \
--print_mask_info \
--csv_file_path csv/gender.csv \




CUDA_VISIBLE_DEVICES=3 python src/validate_on_lfw.py \
--lfw_dir ~/fevemania/datasets/lfw_mtcnnpy_160 \
--model official_checkpoint/careful_pruning_05 \
--lfw_pairs data/custom_pairs.txt \
--lfw_nrof_folds 2 \
--distance_metric 1 \
--subtract_mean \
--use_fixed_image_standardization \
--task_id 1 \
--lfw_batch_size 1 \
--eval_once \
# --use_flipped_images \

# CUDA_VISIBLE_DEVICES=1 python src/train_softmax.py \
# --logs_base_dir logs/facenet/ \
# --models_base_dir official_checkpoint \
# --data_dir ~/fevemania/datasets/CASIA-WebFacealignmtcnn112/ \
# --image_size 160 \
# --model_def models.inception_resnet_v1 \
# --lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160/ \
# --optimizer ADAM \
# --learning_rate -1 \
# --nrof_addiitonal_epochs_to_run 150 \
# --keep_probability 0.8 \
# --random_crop \
# --random_flip \
# --use_fixed_image_standardization \
# --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
# --weight_decay 5e-4 \
# --embedding_size 512 \
# --lfw_distance_metric 1 \
# --lfw_use_flipped_images \
# --lfw_subtract_mean \
# --validation_set_split_ratio 0.05 \
# --validate_every_n_epochs 5 \
# --prelogits_norm_loss_factor 5e-4

tf.get_default_graph().get_tensor_by_name("task_1/Logits/BiasAdd:0")


python src/validate_on_lfw.py \
~/datasets/lfw_mtcnnpy_160 \
original \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization


CUDA_VISIBLE_DEVICES=1 python src/validate_on_lfw.py \
--lfw_dir ~/fevemania/datasets/lfw_mtcnnpy_160 \
--lfw_pairs data/pairs.txt \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization \
--task_id 1 \
--model official_checkpoint/experiment1/gender/test_fold_is_4/model-.ckpt-26 \
--print_mem \
--eval_once \

CUDA_VISIBLE_DEVICES=2 python src/validation.py \
--data_dir '/home/ivclab/fevemania/datasets/age/test_fold_is_4' \
--use_fixed_image_standardization \
--task_name age \
--task_id 2 \
--model official_checkpoint/experiment1/gender/test_fold_is_4/model-.ckpt-26 \
--print_mem \
--eval_once \

CUDA_VISIBLE_DEVICES=3 python src/validation.py \
--data_dir '/home/ivclab/fevemania/datasets/gender/test_fold_is_4' \
--use_fixed_image_standardization \
--task_name gender \
--task_id 3 \
--model official_checkpoint/experiment1/gender/test_fold_is_4/model-.ckpt-26 \
--print_mem \
--eval_once \