#!/bin/bash

# cmmd: bash arithmetic_dataset_noise_ranges.sh 0 12 ; bash arithmetic_dataset_noise_ranges.sh 13 24
# TODO - check ids are correct

id=0 # TODO: id-1

verbose_flag=''
no_save_flag=''

interpolation_ranges=( '[1.1,1.2]' '[0.1,0.2]' '[-1.2,-1.1]' '[-0.2,-0.1]' '[1,2]' '[10,20]' '[-20,-10]' '[-2,-1]' '[-2,2]'           )
extrapolation_ranges=( '[1.2,6]'   '[0.2,2]'   '[-6.1,-1.2]' '[-2,-0.2]'   '[2,6]' '[20,40]' '[-40,-20]' '[-6,-2]' '[[-6,-2],[2,6]]'   )

noise_ranges=( '[0.01,0.05]' '[0.1,0.5]' '[1,2]' '[5,10]' )

for noise_range in "${noise_ranges[@]}"
  do
  # increment id
  id=$((id+1))

  for ((i=0;i<${#interpolation_ranges[@]};++i))
    do
    for seed in $(eval echo {$1..$2})
      do

      export TENSORBOARD_DIR=/data/nalms/tensorboard
      export SAVE_DIR=/data/nalms/saves
      export PYTHONPATH=./

      # NAU-sNMU with mse train loss
      noise_range_prefix=$(echo "${noise_range}" | tr ',' '-' | tr -d \[]) # replace , with - and delete the [ and ]
      experiment_name="FTS_NAU_NMU_ranges/sNMU-noise-ranges/${noise_range_prefix}"
      mkdir -p /data/nalms/logs/${experiment_name}/errors
      python3 -u experiments/simple_function_static.py \
          --id ${id} --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
          --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
          --seed ${seed} --max-iterations 2000000 ${verbose_flag} \
          --name-prefix ${experiment_name} --remove-existing-data --no-cuda --nmu-noise --noise-range ${noise_range} \
          > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
          2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err & # parallel version
    done
    wait
  done
  wait
done
date
echo "Script finished."

