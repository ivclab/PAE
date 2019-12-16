
# bash src/inference_experiment2_task.sh

GPU_ID=0

dataset=("emotion" "chalearn/gender")
MODEL_FOLDER_NAME=("experiment2/emotion/weighted_loss" "experiment2/chalearn/gender/expand_final")
TASK_ID=(2 3)

for idx in `seq 0 1`
do
	echo $idx
	if [ "$idx" = "0" ]
	then
		CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
		--data_dir 'data/'${dataset[idx]} \
		--model 'official_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
		--use_fixed_image_standardization \
		--task_name ${dataset[idx]} \
		--task_id ${TASK_ID[idx]} \
		--eval_once \
		--print_mem \
		--csv_file_path 'accresult/experiment2/'${dataset[idx]}'.csv'
	else
		CUDA_VISIBLE_DEVICES=$GPU_ID python src/validation.py \
		--data_dir 'data/'${dataset[idx]} \
		--model 'official_checkpoint/'${MODEL_FOLDER_NAME[idx]} \
		--use_fixed_image_standardization \
		--task_name ${dataset[idx]} \
		--task_id ${TASK_ID[idx]} \
		--eval_once \
		--print_mem \
		--print_mask_info \
		--csv_file_path 'accresult/experiment2/'${dataset[idx]}'.csv' \
		--share_only_task_1 \
		--filters_expand_ratio 1.2 \
		--history_filters_expand_ratios 1.0,1.2
	fi
done
