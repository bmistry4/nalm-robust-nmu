#!/bin/bash

# cmmd: bash arithmetic_dataset.sh 0 19
# TODO - update path to experiment_name, logging, error logging, interp ranges, extrap ranges
# TODO - make sure log and err log paths exist

verbose_flag=''
no_save_flag=''

interpolation_ranges=( '[1.1,1.2]' '[0.1,0.2]' '[-1.2,-1.1]' '[-0.2,-0.1]' '[1,2]' '[10,20]' '[-20,-10]' '[-2,-1]' '[-2,2]'           )
extrapolation_ranges=( '[1.2,6]'   '[0.2,2]'   '[-6.1,-1.2]' '[-2,-0.2]'   '[2,6]' '[20,40]' '[-40,-20]' '[-6,-2]' '[[-6,-2],[2,6]]'   )

for ((i=0;i<${#interpolation_ranges[@]};++i))
  do
  for seed in $(eval echo {$1..$2})
    do

    export TENSORBOARD_DIR=/data/nalms/tensorboard
    export SAVE_DIR=/data/nalms/saves
    export PYTHONPATH=./

    # TODO - uncomment the relevant experiment and run.
       # NAU-NMU
    python3 -u experiments/simple_function_static.py \
        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
        --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
        --name-prefix FTS_NAU_NMU_ranges/nau-nmu --remove-existing-data --no-cuda \
        > /data/nalms/logs/FTS_NAU_NMU_ranges/nau-nmu/${interpolation_ranges[i]}-${seed}.out \
        2> /data/nalms/logs/FTS_NAU_NMU_ranges/nau-nmu/errors/${interpolation_ranges[i]}-${seed}.err & # parallel version

    # NAU-sNMU with mse train loss
#    python3 -u experiments/simple_function_static.py \
#        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
#        --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#        --seed ${seed} --max-iterations 2000000 ${verbose_flag} \
#        --name-prefix FTS_NAU_NMU_ranges/nau-Nnmu --remove-existing-data --no-cuda --nmu-noise \
#        > /data/nalms/logs/FTS_NAU_NMU_ranges/nau-Nnmu/${interpolation_ranges[i]}-${seed}.out \
#        2> /data/nalms/logs/FTS_NAU_NMU_ranges/nau-Nnmu/errors/${interpolation_ranges[i]}-${seed}.err & # parallel version


    # NAU-batch_sNMU with mse train loss
#    experiment_name='FTS_NAU_NMU_ranges/batch_sNMU'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/simple_function_static.py \
#        --id 7 --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
#        --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#        --seed ${seed} --max-iterations 2000000 ${verbose_flag} \
#        --name-prefix ${experiment_name} --remove-existing-data --no-cuda --nmu-noise --noise-range [1,0] \
#        > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#        2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err & # parallel version

  done
  wait
done
wait
date
echo "Script finished."
