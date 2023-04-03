# Improving the Robustness of Neural Multiplication Units with Reversible Stochasticity
This repository is the official implementation of [Improving the Robustness of Neural Multiplication Units with Reversible Stochasticity](https://arxiv.org/abs/2211.05624).

This work builds ontop of the research on **[Neural Arithmetic Units](https://openreview.net/forum?id=H1gNOeHKPS) 
by Andreas Madsen and Alexander Rosenberg Johansen**. 
The [original code](https://github.com/AndreasMadsen/stable-nalu) is by Andreas Madsen, who created the 
underlying framework used to create datasets, run experiments, and generate plots. 
**See their original README ([below](#neural-arithmetic-units))** (which includes requirements).

## About 
Neural Arithmetic Logic Modules are differentiable networks which can learn to do arithmetic/logic in an extrapolative
manner with the by-product of having interpretable weights. 
However learning the ideal weights for such modules can be challenging, with modules lacking robustness to different training
distributions. 
Our work focuses on taking a multiplication NALM - the [Neural Multiplication Unit](https://openreview.net/forum?id=H1gNOeHKPS)
and applying a form of reversible stochasticity to improve its robustness. We call this module the sotchasitic NMU (sNMU). 

## Setup env
Generate a conda environment called nalu-env: 
`conda env create -f nalu-env.yml`

Install stable-nalu:
`python3 setup.py develop`

## Recreating Experiments From the Paper

### Single Module Task (Figures 1 and 2)
First, create a csv file containing the threshold values for each range using 
<pre> Rscript <a href="export/single_layer_task/benchmark/generate_exp_setups.r">generate_exp_setups.r</a> </pre>

#### Generating plots consists of 3 stages
1. Run a shell script which calls the python script to _generate the tensorboard results_ over multiple seeds and ranges
    - `bash lfs_batch_jobs/icml_2022/single_module.sh 0 24`
    - The *0 24* will run 25 seeds in parallel (i.e. seeds 0-24). 
    
2. Call the python script to convert the tensorboard _results to a csv file_
    - `python3 export/simple_function_static.py --tensorboard-dir 
/data/nalms/tensorboard/<experiment_name>/ --csv-out /data/nalms/csvs/<experiment_name>.csv`
        - `--tensorboard-dir`: Directory containing the tensorboard folders with the model results
        - `--csv-out`: Filepath on where to save the csv result file
        - `<experiment_name>`: value of the experiment_name variable in the shell script used for step 1

3. Call the R script(s) to convert the csv results to a _plot_ (saved as pdf)
    - **MLP vs NALM surface plots (Figure 1):** 
        - Generate gold results: 
        <pre> python3 <a href="export/single_layer_task/benchmark/pretrained_model_prediction_sweep.py">pretrained_model_prediction_sweep.py</a> 
        --csv-save-folder /data/nalms/plots/2d-surface-plots/csvs --gold-outputs </pre>
        
        - Load saved model and save it's prediction over the grid points: 
        <pre>python3 <a href="export/single_layer_task/benchmark/pretrained_model_prediction_sweep.py">pretrained_model_prediction_sweep.py</a>  
        --layer-type ReRegualizedLinearMNAC 
        --model-filepath /data/nalms/saves/&lt;experiment_name&gt/&lt;checkpoint_filename&gt;.pth 
        --csv-save-folder/data/nalms/plots/2d-surface-plots/csvs 
        --csv-save-filename &lt;RESULTS FILENAME (your choice) (excluding .csv)&gt; </pre>
        
        - Plot surface map: 
        <pre> Rscript <a href="export/single_layer_task/benchmark/plot_two_input_surface.r">plot_two_input_surface.r</a> 
        /data/nalms/plots/2d-surface-plots/csvs/ /data/nalms/plots/2d-surface-plots/2D-surface-mul mul </pre>
    
    - **Single Module Task for all NALMs (Figure 2):**
        <pre> Rscript <a href="export/single_layer_task/benchmark/plot_results.r">plot_results.r</a> 
        /data/nalms/csvs/&lt;experiment_name&gt data/nalms/plots/benchmark/sltr-in2/ benchmark_sltr op-mul None benchmark_sltr_mul </pre>
            - First arg: N/A
            - Second arg: Path to directory where you want to save the plot file
            - Third arg: Contributes to the plot filename. Use the Output value (see table below).
            - Forth arg: Arithmetic operation to create plot. Use op-mul.
            - Fifth arg: N/A
            - Sixth arg: Lookup key used to load relevant files and plot information

### Arithmetic Dataset Task (Figures 5 and 7) 
Create a csv file containing the threshold values for each range using <pre> Rscript <a href="export/function_task_static_ranges/generate_exp_setups.r">generate_exp_setups.r</a> </pre>

1. Run:
    - `bash lfs_batch_jobs/icml_2022/arithmetic_dataset.sh 0 19` 
        - Uncomment the model you want to run in the script 
        - You may need to change the values for: TENSORBOARD_DIR, SAVE_DIR, and the .err and .log filepaths to work with your local filesystem
    - `bash lfs_batch_jobs/icml_2022/arithmetic_dataset_noise_ranges.sh 0 12 ; bash lfs_batch_jobs/icml_2022/arithmetic_dataset_noise_ranges.sh 13 24`
        
2. `python3 export/simple_function_static.py --tensorboard-dir 
/data/nalms/tensorboard/FTS_NAU_NMU_ranges/<experiment_name> --csv-out /data/nalms/csvs/<experiment_name>.csv`
    - `--tensorboard-dir`: Directory containing the tensorboard folders with the model results
    - ` --csv-out`: Filepath on where to save the csv result file
    
3. 
    - Figure 5: <pre> Rscript <a href="export/function_task_static_ranges/plot_fts_results.r">plot_fts_results.r</a> /data/nalms/csvs/ /data/nalms/plots/FTS_NAU_NMU_ranges/ fts-snmu-noise-ranges fts-snmu-noise-ranges </pre>
    - Figure 7: <pre> Rscript <a href="export/function_task_static_ranges/plot_fts_results.r">plot_fts_results.r</a> /data/nalms/csvs/ /data/nalms/plots/FTS_NAU_NMU_ranges/ fts-2021-final fts-2021-final </pre>
    
### Static MNIST Product (Figures 8 and 9)
1. Run experiment script to generate tensorboard results: `bash lfs_batch_jobs/icml_2022/static_mnist_prod_isolated.sh 0 9 <GPU ID>` 
and `bash lfs_batch_jobs/icml_2022/static_mnist_prod_colour_concat.sh 0 2 <GPU ID>`
2. Convert tensorboard results to csvs: `python3 export/two_digit_mnist/two_digit_mnist_reader.py --tensorboard-dir /data/nalms/tensorboard/<experiment_name> --csv-out /data/nalms/csvs/static-mnist/mul/<model_name>.csv`

3. Create the plot:
    - Isolated digits setup (Figure 8): <pre> Rscript <a href="export/two_digit_mnist/plot_results.r">plot_results.r</a> /data/nalms/csvs/static-mnist/mul/ /data/nalms/plots/static-mnist/1digit_conv/ 2DMNIST-static-mnist-mul-isolated None static-mnist-mul-isolated </pre>
    - Colour channel concatenated setup (Figure 9): <pre> Rscript <a href="export/two_digit_mnist/plot_results.r">plot_results.r</a> /data/nalms/csvs/static-mnist/mul/ /data/nalms/plots/static-mnist/MSE_Adam-lr0.001_TPS-no-concat-conv/ 2DMNIST-static-mnist-mul-colour-concat None static-mnist-mul-colour-concat </pre>

### Sequential MNIST Product (Figure 10)
1. Generate tensorboard results for the reference model and comparison models:
    - Reference: `bash lfs_batch_jobs/icml_2022/sequential_mnist_prod_reference.sh`
    - Comparison: `bash lfs_batch_jobs/icml_2022/sequentialmnist_prod_long.sh 0 9  <GPU ID>`

2. Convert tensorboard results to csvs:
    - Reference: `python3 export/sequential_mnist.py --tensorboard-dir /data/nalms/tensorboard/sequential_mnist/sequential_mnist_prod_reference/ --csv-out  /data/nalms/csvs/sequential_mnist_prod_reference.csv`
    - Comparison: `python3 export/sequential_mnist.py --tensorboard-dir /data/nalms/tensorboard/sequential_mnist/<exoeriment_name> --csv-out  /data/nalms/csvs/sequential_mnist_prod_long_<MODEL NAME>.csv`
        - Update `--tensorboard-dir` and `--csv-out` for each module
3. Create the plot: <pre> Rscript <a href="export/sequential_mnist/sequential_mnist_prod_long.r">sequential_mnist_prod_long.r</a> /data/nalms/csvs/ /data/nalms/plots/sequential_mnist_prod/ sequential_mnist_prod_long</pre>

## Appendix G (Figures 13-20)
This section assumes that the experiments for the Static MNIST Product have been run. (See above for instructions on how 
to run the experiments.)

### Weight Trajectory figures:
Run: 
`Rscript weights_path.r /data/nalms/csvs/<experiment_name> /data/nalms/plots/weight-paths/ <LOAD CSV FILENAME (excluding the .csv)> _labels2out-path`

### Class accuracy and confusion matrix plots:
1. Open the python file for the relevant task in the `/experiments/` folder
2. Search for the 'PRETRAINED MODEL VISUALISATION CODE' section and uncomment the loading code
3. Set the `load_filename` variable to the filepath for the saved checkpoint (ending in .pth) 
4. Set the relevant arguments for the model
3. Run the python script which will save the confusion matrix (and the per-label accuracy csv). 
4. Open <a href="export/two_digit_mnist/plot_per_class_accuracies.r">plot_per_class_accuracies.r</a> and set the `merge_mode` to either 
`mul-1digit_conv-Adam` to create Figure 15 or `MSE_Adam-lr0.001_TPS-no-concat-conv_ROUNDED` to create Figure 17. 
5. Run the Rscript which will plot the class accuracies. 

### Epoch vs label accuracy figures:
Generated in step 3 of [Static MNIST Product](#static-mnist-product-figures-8-and-9)

---
# Neural Arithmetic Units

This code encompass two publiations. The ICLR paper is still in review, please respect the double-blind review process.

![Hidden Size results](readme-image.png)

_Figure, shows performance of our proposed NMU model._

## Publications

#### SEDL Workshop at NeurIPS 2019

Reproduction study of the Neural Arithmetic Logic Unit (NALU). We propose an improved evaluation criterion of arithmetic tasks including a "converged at" and a "sparsity error" metric. Results will be presented at [SEDL|NeurIPS 2019](https://sites.google.com/view/sedl-neurips-2019/#h.p_vZ65rPBhIlB4). – [Read paper](http://arxiv.org/abs/1910.01888).

```bib
@inproceedings{maep-madsen-johansen-2019,
    author={Andreas Madsen and Alexander Rosenberg Johansen},
    title={Measuring Arithmetic Extrapolation Performance},
    booktitle={Science meets Engineering of Deep Learning at 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)},
    address={Vancouver, Canada},
    journal={CoRR},
    volume={abs/1910.01888},
    month={October},
    year={2019},
    url={http://arxiv.org/abs/1910.01888},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    eprint={1910.01888},
    timestamp={Fri, 4 Oct 2019 12:00:36 UTC}
}
```

#### ICLR 2020 (Under review)

Our main contribution, which includes a theoretical analysis of the optimization challenges with the NALU. Based on these difficulties we propose several improvements. **This is under double-blind peer-review, please respect our anonymity and reference https://openreview.net/forum?id=H1gNOeHKPS and not this repository!** – [Read paper](https://openreview.net/forum?id=H1gNOeHKPS).

```bib
@inproceedings{mnu-madsen-johansen-2020,
    author={Andreas Madsen and Alexander Rosenberg Johansen},
    title={Neural Arithmetic Units},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=H1gNOeHKPS},
    note={under review}
}
```

## Install

```bash
python3 setup.py develop
```

This will install this code under the name `stable-nalu`, and the following dependencies if missing: `numpy, tqdm, torch, scipy, pandas, tensorflow, torchvision, tensorboard, tensorboardX`.

## Experiments used in the paper

All experiments results shown in the paper can be exactly reproduced using fixed seeds. The `lfs_batch_jobs`
directory contains bash scripts for submitting jobs to an LFS queue. The `bsub` and its arguments, can be
replaced with `python3` or an equivalent command for another queue system.

The `export` directory contains python scripts for converting the tensorboard results into CSV files and
contains R scripts for presenting those results, as presented in the paper.

## Naming changes

As said earlier the naming convensions in the code are different from the paper. The following translations
can be used:

* Linear: `--layer-type linear`
* ReLU: `--layer-type ReLU`
* ReLU6: `--layer-type ReLU6`
* NAC-add: `--layer-type NAC`
* NAC-mul: `--layer-type NAC --nac-mul normal`
* NAC-sigma: `--layer-type PosNAC --nac-mul normal`
* NAC-nmu: `--layer-type ReRegualizedLinearPosNAC --nac-mul normal --first-layer ReRegualizedLinearNAC`
* NALU: `--layer-type NALU`
* NAU: `--layer-type ReRegualizedLinearNAC`
* NMU: `--layer-type ReRegualizedLinearNAC --nac-mul mnac`

## Extra experiments

Here are 4 experiments in total, they correspond to the experiments in the NALU paper.

```
python3 experiments/simple_function_static.py --help # 4.1 (static)
python3 experiments/sequential_mnist.py --help # 4.2
```

Example with using NMU on the multiplication problem:

```bash
python3 experiments/simple_function_static.py \
    --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
    --seed 0 --max-iterations 5000000 --verbose \
    --name-prefix test --remove-existing-data
```

The `--verbose` logs network internal measures to the tensorboard. You can access the tensorboard with:

```
tensorboard --logdir tensorboard
```
