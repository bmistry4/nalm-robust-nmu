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
  experiment_name='static-mnist/mul/1digit_conv-snmu'
  mkdir -p ${logs_base_dir}/${experiment_name}/errors
  id=1
  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/two_digit_mnist.py \
    --seed ${seed} \
    --id ${id} --operation mul --use-nalm --learn-labels2out --nmu-noise --max-epochs 1000 \
    --regualizer-scaling-start 30 --regualizer-scaling-end 40  --regualizer 100 \
    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
    > ${logs_base_dir}/${experiment_name}/${seed}.out \
    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err   

#  experiment_name='static-mnist/mul/1digit_conv-mul'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=2
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/two_digit_mnist.py \
#    --seed ${seed} \
#    --id ${id} --operation mul --max-epochs 1000 \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

#  experiment_name='static-mnist/mul/1digit_conv-nmu'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=3
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/two_digit_mnist.py \
#    --seed ${seed} \
#    --id ${id} --operation mul --use-nalm --learn-labels2out --max-epochs 1000 \
#    --regualizer-scaling-start 30 --regualizer-scaling-end 40  --regualizer 100 \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

#  experiment_name='static-mnist/mul/1digit_conv-fc'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=4
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/two_digit_mnist.py \
#    --seed ${seed} \
#    --id ${id} --operation mul --learn-labels2out --max-epochs 1000 \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

#  experiment_name='static-mnist/mul/1digit_conv-batch-snmu'
#  mkdir -p ${logs_base_dir}/${experiment_name}/errors
#  id=5
#  CUDA_VISIBLE_DEVICES=$3 python3 -u experiments/two_digit_mnist.py \
#    --seed ${seed} \
#    --id ${id} --operation mul --use-nalm --learn-labels2out --nmu-noise --noise-range [1,0] --max-epochs 1000 \
#    --regualizer-scaling-start 30 --regualizer-scaling-end 40 --regualizer 100 \
#    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 4  \
#    > ${logs_base_dir}/${experiment_name}/${seed}.out \
#    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err
    
done
wait 

date
echo "Script finished."
