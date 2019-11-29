
# bash src/inference_experiment1_task.sh

GPU_ID=0

dataset=("age/test_fold_is_0" "gender/test_fold_is_0" "age/test_fold_is_1" "gender/test_fold_is_1" "age/test_fold_is_2" "gender/test_fold_is_2" "age/test_fold_is_3" "gender/test_fold_is_3" "age/test_fold_is_4" "gender/test_fold_is_4")
MODEL_FOLDER_NAME=("experiment1/age/test_fold_is_0" "experiment1/gender/test_fold_is_0" "experiment1/age/test_fold_is_1" "experiment1/gender/test_fold_is_1" "experiment1/age/test_fold_is_2" "experiment1/gender/test_fold_is_2" "experiment1/age/test_fold_is_3" "experiment1/gender/test_fold_is_3" "experiment1/age/test_fold_is_4" "experiment1/gender/test_fold_is_4")
TASK_ID=(2 3 2 3 2 3 2 3 2 3)

for idx in `seq 0 1`
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
    --data_dir 'data/'${dataset[idx]} \
    --model 'official_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
    --use_fixed_image_standardization \
    --task_name ${dataset[idx]} \
    --task_id ${TASK_ID[idx]} \
    --eval_once \
    --print_mem \
    --csv_file_path 'accresult/experiment1/'${dataset[idx]}'.csv'
done
