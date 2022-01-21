#!/bin/bash

# cmmd: bash minerva_sequential_mnist_prod_reference.sh 
# TODO - update path to experiment_name, logging, error logging, interp ranges, extrap ranges
# TODO - make sure log and err log paths exist
# TODO check gpu is available

experiment_name='sequential_mnist/sequential_mnist_prod_reference'
verbose_flag=''

for seed in {0..9}
do
  
  export TENSORBOARD_DIR=/data/nalms/tensorboard
  export SAVE_DIR=/data/nalms/saves
  export PYTHONPATH=./
  
  CUDA_VISIBLE_DEVICES=0 python3 -u experiments/sequential_mnist.py \
    --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 123456789 \
    --model-simplification solved-accumulator \
    --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
    --seed ${seed} --max-epochs 1000 \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5 \
    > /data/nalms/logs/sequential_mnist/sequential_mnist_prod_reference/${seed}.out \
    2> /data/nalms/logs/sequential_mnist/sequential_mnist_prod_reference/errors/${seed}.err
     
done

date
echo "Script finished."