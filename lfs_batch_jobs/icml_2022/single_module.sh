#!/bin/bash

# cmmd: bash single_module.sh 0 24
export LSB_JOB_REPORT_MAIL=N


verbose_flag=''
no_save_flag='--no-save'
log_interval='1000'

# NOTE: the ReLU runs only require running interpolation range [1,2] and extrapolation range [2,6].
interpolation_ranges=( '[-20,-10]' '[-2,-1]' '[-1.2,-1.1]' '[-0.2,-0.1]' '[-2,2]'          '[0.1,0.2]' '[1,2]' '[1.1,1.2]' '[10,20]' )
extrapolation_ranges=( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )

for ((i=0;i<${#interpolation_ranges[@]};++i))
  do
  for seed in $(eval echo {$1..$2})
    do

    export TENSORBOARD_DIR=/data/nalms/tensorboard
    export SAVE_DIR=/data/nalms/saves
    export PYTHONPATH=./


#######################################################################################################################
# MULTIPLICATION
#######################################################################################################################
# TODO - uncomment the relevant experiment and run.
################# MLPs #################
    experiment_name='benchmark/sltr-in2/mul/ReLU-h1'
    mkdir -p /data/nalms/logs/${experiment_name}/errors
    python3 -u experiments/single_layer_benchmark.py \
    --operation mul --layer-type ReLU --hidden-size 1 --regualizer 0 --mlp-bias \
    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
    --name-prefix ${experiment_name} --remove-existing-data --no-cuda \
    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/ReLU-h100'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type ReLU --hidden-size 100 --regualizer 0 --mlp-bias \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 2000000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &
########################################
#    experiment_name='benchmark/sltr-in2/NMU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type ReRegualizedLinearMNAC --nac-mul mnac \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/MNAC'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type MNAC --nac-mul normal \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/NALU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type NALU \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/NPU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type NPU --nac-mul npu \
#    --learning-rate 5e-3 --regualizer-beta-start 1e-7 --regualizer-beta-end 1e-5 \
#    --regualizer-l1 --regualizer-shape none --regualizer 0 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/RealNPU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type RealNPU --nac-mul real-npu \
#    --learning-rate 5e-3 --regualizer-beta-start 1e-7 --regualizer-beta-end 1e-5 \
#    --regualizer-l1 --regualizer-shape none --regualizer 0 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/iNALU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type iNALU \
#    --clip-grad-value 0.1 --reinit \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/G-NALU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --operation mul --layer-type NALU \
#    --nalu-gate golden-ratio --nalu-mul golden-ratio --nac-weight golden-ratio \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/mul/batch-sNMU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --id 6 --operation mul --layer-type ReRegualizedLinearMNAC --nac-mul mnac --nmu-noise \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

  done
  wait
done
wait
date
echo "Script finished."
