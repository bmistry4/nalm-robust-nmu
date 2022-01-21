#!/bin/bash

# cmmd: bash sequential_mnist_prod_long.sh 0 4 <GPU-ID> & bash sequential_mnist_prod_long.sh 5 9 <GPU-ID>

# TODO - make sure log and err log paths exist
# TODO check gpu is available

logs_base_dir='/data/nalms/logs'
verbose_flag='--verbose'

# do 2 runs on 2 gpus - seeds 0-4 and 0-9
for seed in $(eval echo {$1..$2})
do
    export TENSORBOARD_DIR=/data/nalms/tensorboard
    export SAVE_DIR=/data/nalms/saves
    export PYTHONPATH=./

    # NMUCell
    experiment_name='sequential_mnist/sequential_mnist_prod_long/nmu'
    mkdir -p ${logs_base_dir}/${experiment_name}/errors
    CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/sequential_mnist.py \
      --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
      --mnist-digits 123456789 --mnist-outputs 1 \
      --regualizer-z 1 \
      --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
      --seed ${seed} --max-epochs 1000 ${verbose_flag} \
      --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5 \
      > ${logs_base_dir}/${experiment_name}/${seed}.out \
      2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

    # sNMUCell with mse train loss
#    experiment_name='sequential_mnist/sequential_mnist_prod_long/snmu'
#    mkdir -p ${logs_base_dir}/${experiment_name}/errors
#    CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/sequential_mnist.py \
#      --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
#      --mnist-digits 123456789 --mnist-outputs 1 \
#      --regualizer-z 1 \
#      --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
#      --seed ${seed} --max-epochs 1000 ${verbose_flag} \
#      --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5 --nmu-noise \
#      > ${logs_base_dir}/${experiment_name}/${seed}.out \
#      2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

#     sNMUCell (with noise range [1, 0] for batch stats) with mse train loss
#    experiment_name='sequential_mnist/sequential_mnist_prod_long/batch-snmu'
#    mkdir -p ${logs_base_dir}/${experiment_name}/errors
#    CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/sequential_mnist.py \
#      --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
#      --mnist-digits 123456789 --mnist-outputs 1 \
#      --regualizer-z 1 \
#      --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
#      --seed ${seed} --max-epochs 1000 ${verbose_flag} \
#      --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5 --nmu-noise --noise-range [1,0] \
#      > ${logs_base_dir}/${experiment_name}/${seed}.out \
#      2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err
done

date
echo "Script finished."

