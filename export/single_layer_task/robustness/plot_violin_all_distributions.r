rm(list = ls())
#setwd('C:/Users/mistr/Documents/SOTON/PhD/Code/nalu-stable-exp/export/single_layer_task/robustness/')

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
model_name <- args[3] # name of model to use in plot (i.e. short name). Use 'None' if you don't want to change the default model name. To be used on results file with only one model.

library(ggplot2)
library(plyr)
library(dplyr)
library(readr)
library(tibble)

source('./weight_distances.r')
source('./_robustness_expand_name.r')
source('./generate_solutions.r')

csv_merger = function(load_files_names, distributions_list) {
  combined_tables <- NULL
  # load tables for each element in the list
  tables <- lapply(load_files_names, read_csv)
  for (idx in 1:length(tables)) {
    t <- ldply(tables[idx], data.frame)  # convert from list to df
    # don't process dfs with no rows - to avoid dists where all configs failed to reach required max step
    if (!empty(t)){
      t$distribution.id <- distributions_list[[idx]]      # rename the model name to pre-defined value in list
      combined_tables <- rbind(combined_tables, t)  # add model data to an accumulated table
    }
  }
  return(combined_tables)
}

calculate_param_dists = function(model_table, gold_table) {
  distance_df = data.frame(
    distribution.id = character(), model = character(), seed = integer(), distance = numeric(),
    train.distribution = character(), interpolation.range = character()
  )

  # merge golden solutions into
  model_table = merge(dat, gold_table, by = c('model', 'operation', 'seed'))
  # go through each different experiment config
  for (i in 1:nrow(model_table)) {
    min_dist = lookup_model_distance(model_table[i,])
    distance_df = add_row(distance_df,
        distribution.id = model_table[i,]$distribution.id,
        model = model_table[i,]$model, seed = model_table[i,]$seed, distance = min_dist,
        train.distribution = model_table[i,]$train.distribution, interpolation.range = model_table[i,]$interpolation.range
    )
  }
  return(distance_df)
}


# SETUP LOAD AND SAVE FILES NAMES
#load_file = paste0(load_folder, base_filename, '.csv')
#load_file = paste0(load_folder, base_filename, '_expanded_100K_step.csv')
base_filename = 'distributions'
base_output = paste0(results_folder, base_filename)
violin_output = paste0(base_output, '_violin.pdf')
#dat = read_csv(load_file)
#write.csv(filter.by.step(expand.name(dat)), paste0(load_folder, base_filename, '_expanded_100K_step.csv'))

# Toy experiments setup
#model_name = 'None' # 'Real NPU (mod)'
#load_file = 'hparam_search_nau-add_truncated-normal-0-1--5-5.csv' # 'nmru-div_truncated-normal-0-1--5-5_expanded_last_step.csv' # 'nmru-div_truncated-normal-0-1--5-5_expanded_last_step.csv'#
#violin_output = 'violin_plot.pdf'
#dat = filter.by.step(expand.name(read_csv(load_file)))  # TODO: use if NAU-add
#dat = read_csv(load_file)
#write.csv(filter.by.step(expand.name(dat)), paste0(load_folder, base_filename, '_expanded_100K_step.csv'))
#dat$distribution.id = 'test' # TODO use if toy exp

dat = csv_merger(list(
  paste0(load_folder, 'truncated-normal-0-1--5-5', '_expanded_100K_step.csv'),
  paste0(load_folder, 'uniform-1-2', '_expanded_100K_step.csv'),
  paste0(load_folder, 'uniform--2--1', '_expanded_100K_step.csv'),
  paste0(load_folder, 'exponential-500-0-0', '_expanded_100K_step.csv')
),
  list('truncated normal [-5,5]', 'uniform [1,2]', 'uniform [-2,-1]', 'exponential 500')
)

# Rename model with the given arg
if (model_name != 'None') {
  dat$model = model_name
}

coarseness = 'all-distributions'
golden_solutions = create_golden_solutions()
distances_dat = calculate_param_dists(dat, golden_solutions)
distances_dat = distances_dat %>% group_data_by_coarseness(coarseness)

violin_plot = plot_violin(distances_dat)
#violin_plot

ggsave(violin_output, violin_plot, device = "pdf")
write.csv(distances_dat, paste0(base_output, '_violin_plot_data.csv'))

print("R Script completed.")
