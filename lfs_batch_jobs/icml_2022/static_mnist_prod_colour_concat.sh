#!/bin/bash

# cmmd: bash run.sh <start seed> <end seed> <GPU-ID>
# loops over the different folds

logs_base_dir='/data/nalms/logs'
data_path_flag='/data/nalms/datasets/static-mnist'
no_cuda_flag=''
no_save_flag=''

for seed in $(eval echo {$1..$2})
  do

  export TENSORBOARD_DIR=/data/nalms/tensorboard
  export SAVE_DIR=/data/nalms/saves
  export PYTHONPATH=./
  
# TODO - uncomment the relevant experiment and run.
  experiment_name='static-mnist/mul/mul_MSE_Adam-lr0.001_TPS-no-concat-conv'
  mkdir -p ${logs_base_dir}/${experiment_name}/errors
  id=6
  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/ST_mnist_labels.py \
    --seed ${seed} \
    --num-folds 3 --batch-size 256 --learning-rate 0.001 --img2label-model TPS-no-concat-conv --optimizer adam --no-scheduler \
    --id ${id} --operation mul --max-epochs 1000 --scheduler-step-size 2000 --samples-per-permutation 1100 \
    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
    > ${logs_base_dir}/${experiment_name}/${seed}.out \
    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

  experiment_name='static-mnist/mul/snmu_MSE_Adam-lr0.001_TPS-no-concat-conv'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=7
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/ST_mnist_labels.py \
#    --seed ${seed} \
#    --num-folds 3 --batch-size 256 --learning-rate 0.001 --img2label-model TPS-no-concat-conv --optimizer adam --no-scheduler \
#    --id ${id} --operation mul --max-epochs 1000 --scheduler-step-size 2000 --samples-per-permutation 1100 \
#    --use-nalm --learn-labels2out --nmu-noise \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

#  experiment_name='static-mnist/mul/nmu_MSE_Adam-lr0.001_TPS-no-concat-conv'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=8
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/ST_mnist_labels.py \
#    --seed ${seed} \
#    --num-folds 3 --batch-size 256 --learning-rate 0.001 --img2label-model TPS-no-concat-conv --optimizer adam --no-scheduler \
#    --id ${id} --operation mul --max-epochs 1000 --scheduler-step-size 2000 --samples-per-permutation 1100 \
#    --use-nalm --learn-labels2out \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

#  experiment_name='static-mnist/mul/fc_MSE_Adam-lr0.001_TPS-no-concat-conv'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=9
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/ST_mnist_labels.py \
#    --seed ${seed} \
#    --num-folds 3 --batch-size 256 --learning-rate 0.001 --img2label-model TPS-no-concat-conv --optimizer adam --no-scheduler \
#    --id ${id} --operation mul --max-epochs 1000 --scheduler-step-size 2000 --samples-per-permutation 1100 \
#    --learn-labels2out \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

#  experiment_name='static-mnist/mul/batch-snmu_MSE_Adam-lr0.001_TPS-no-concat-conv'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=10
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/ST_mnist_labels.py \
#    --seed ${seed} \
#    --num-folds 3 --batch-size 256 --learning-rate 0.001 --img2label-model TPS-no-concat-conv --optimizer adam --no-scheduler \
#    --id ${id} --operation mul --max-epochs 1000 --scheduler-step-size 2000 --samples-per-permutation 1100 \
#    --use-nalm --learn-labels2out --nmu-noise --noise-range [1,0] \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

done
wait

date
echo "Script finished."

