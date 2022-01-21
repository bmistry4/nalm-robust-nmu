rm(list = ls())
#setwd('C:/Users/mistr/Documents/SOTON/PhD/Code/nalu-stable-exp/export/single_layer_task/robustness/')

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. nau-add_truncated-normal-0-1--5-5.csv). If set to None it will sweep over multiple distributions.
model_name <- args[4] # name of model to use in plot (i.e. short name). Use 'None' if you don't want to change the default model name. To be used on results file with only one model.

library(ggplot2)
library(plyr)
library(dplyr)
library(readr)
library(tibble)

source('./weight_distances.r')
source('./_robustness_expand_name.r')
source('./generate_solutions.r')

calculate_param_dists = function(model_table, gold_table) {
  distance_df = data.frame(
    config.id = character(), model = character(), seed = integer(), distance = numeric(),
    train.distribution = character(), interpolation.range = character()
  )

  # merge golden solutions into
  # FIXME: this part causes issue for NAU
  model_table = merge(dat, gold_table, by = c('model', 'operation', 'seed'))
  # go through each different experiment config
  for (i in 1:nrow(model_table)) {
    min_dist = lookup_model_distance(model_table[i,])
    distance_df = add_row(distance_df,
                          config.id = generate_config_id(model_table[i,]),
                          model = model_table[i,]$model, seed = model_table[i,]$seed, distance = min_dist,
                          train.distribution = model_table[i,]$train.distribution, interpolation.range = model_table[i,]$interpolation.range
    )
  }
  return(distance_df)
}

# create list to process either a single file (i.e. one distribution) or multiple files
if (base_filename == 'None') {
  distributions_list = list('truncated-normal-0-1--5-5', 'uniform-1-2', 'uniform--2--1', 'exponential-500-0-0')
} else {
  distributions_list = list(base_filename)
}

for (current_distribution in distributions_list) {
  base_filename = current_distribution

  # SETUP LOAD AND SAVE FILES NAMES
  load_file = paste0(load_folder, base_filename, '_expanded_100K_step', '.csv')
  #load_file = paste0(load_folder, base_filename, '.csv')
  base_output = paste0(results_folder, base_filename)
  ci_output = paste0(base_output, '_ci.pdf')

  # Toy experiments setup
  #model_name = 'None' # 'Real NPU (mod)'
  #load_file = 'nmru-div_truncated-normal-0-1--5-5_expanded_last_step.csv' # 'hparam_search_nau-add_truncated-normal-0-1--5-5.csv' #  'nmru-div_truncated-normal-0-1--5-5_expanded_last_step.csv'#
  #ci_output = 'ci_plot.pdf'

  # TODO: use when creating plots. Do not use if generating the _expanded_100K_step files
  dat = read_csv(load_file) # USE IF NOT SAVING THE _expanded file version
  # TODO: use to generate the '<XX>_expanded_100K_step.csv' file
  #dat = read_csv(paste0(load_folder, base_filename, '.csv'))
  #write.csv(filter.by.step(expand.name(dat)), paste0(load_folder, base_filename, '_expanded_100K_step.csv'))

  # Rename model with the given arg
  if (model_name != 'None') {
    dat$model = model_name
  }
  
  coarseness = 'config'
  golden_solutions = create_golden_solutions()
  distances_dat = calculate_param_dists(dat, golden_solutions)
  distances_dat = distances_dat %>% group_data_by_coarseness(coarseness)

  distances_summary_dat = distances_summary(distances_dat)
  ci_plot = plot_scatter_with_ci(distances_summary_dat)
#  ci_plot

  ggsave(ci_output, ci_plot, device = "pdf")
  write.csv(distances_summary_dat, paste0(base_output, '_ci_plot_data.csv'))

  print("R Script completed.")
}
