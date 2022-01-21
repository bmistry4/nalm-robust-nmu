# This script will output the stats of an exp run over multiple seeds summarising the mean, sd, se, 95% CI for the interp error (val), extrap error (test), sparsity error and best iteration step.

rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results 
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
op <- args[4] # operation filter on (e.g. op-mul)
model_name <- args[5] # name of model to use in plot (i.e. short name). Use 'None' if you don't want to change the default model name. To be used on results file with only one model.
model_name=ifelse(is.na(model_name), 'None', model_name)  # no passed arg becomes 'None' i.e. use default name
merge_mode <- args[6] # if 'None' then just loads single file. Otherwise looks up multiple results to merge together (use when have multiple models to plot)
merge_mode=ifelse(is.na(merge_mode), 'None', merge_mode)  # no passed arg becomes 'None' i.e. single model plot
parameter_value <- args[7]  # type of experiment e.g. extrapolation.ranges (see exp_setups for value options)
parameter_value=ifelse(is.na(parameter_value), 'extrapolation.range', parameter_value) # is no argument given then assume you want extrapolation.range
csv_ext = '.csv'

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)
source('../_single_layer_task_expand_name.r')
source('../../_compute_summary.r')
source('../../_plot_parameter.r')
source('./_table.r')
source('./csv_merger.r')

data_summary = function(df, metric_col_name, metric_type) {
  # 95% confidence intervals using a t-test
  return(df %>%
           summarize(n = n(),
                     mean = mean(!!metric_col_name),
                     sd = sd(!!metric_col_name),
                     se = sd / sqrt(n),
                     ci = qt(0.975, df = n - 1) * sd / sqrt(n),
           ) %>%
           mutate(metric.type = metric_type)
  )
}

# number of logged steps to look at (starting from the last step and working backwards).
best.range = 5000

# find the step with the lowest error in the allowed range of steps.
# a range larger than the errors length will just consider all the rows in errors
best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

# return the first step where the error is under the allowed threshold. If none exists, return NA.
first.solved.step = function (steps, errors, threshold) {
  index = first(which(errors < threshold))
  if (is.na(index)) {
    return(NA)
  } else {
    return(steps[index])
  }
}

name.parameter = 'interpolation.range'  # column name containing x-axis values
name.label = 'Interpolation range'      # x-axis label
name.file = paste0(load_folder, base_filename, csv_ext)
name.output = paste0(results_folder, base_filename, '_', op)

# load the experiment setups file containing the thresholds and filter for the relevant experiments.
eps = read_csv('./exp_setups.csv') %>%
  filter(simple == FALSE & parameter == parameter_value & operation == op) %>%
  mutate(
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, extrapolation.range, epsilon)

# load (and merge) the exp results csvs and merge with the experument setup files
dat = load.and.merge.csvs(merge_mode)  %>%
  # to maintain ordering of dat use join not merge (otherwise the solved at subplot will be incorrect)
  inner_join(eps)  %>%
  mutate(
    # !! = remember the expression I stored recently? Now take it, and ‘unquote’ it, that is, just run it!”
    parameter = !!as.name(name.parameter)
  )

dat.last = dat %>%
  group_by(name, parameter) %>%
  summarise(
    threshold = last(epsilon),
    best.model.step = best.model.step.fn(metric.valid.interpolation),
    interpolation.last = metric.valid.interpolation[best.model.step],
    extrapolation.last = metric.test.extrapolation[best.model.step],
    interpolation.step.solved = first.solved.step(step, metric.valid.interpolation, threshold),
    extrapolation.step.solved = first.solved.step(step, metric.test.extrapolation, threshold),
    sparse.error.max = sparse.error.max[best.model.step],
    solved.extrapolation = replace_na(metric.test.extrapolation[best.model.step] < threshold, FALSE),
    solved.interpolation = replace_na(metric.valid.interpolation[best.model.step] < threshold, FALSE),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n(),
    best.iteration = step[best.model.step]
  )

dat.last = dat.last %>% group_by(model, parameter)

# Rename model with the given arg (use for single file processing)
if (model_name != 'None') {
  dat.last$model = model_name
}
  
summary_dat_interp = data_summary(dat.last, quo(interpolation.last), 'interpolation mse')
summary_dat_extrap = data_summary(dat.last, quo(extrapolation.last), 'extrapolation mse')
summary_dat_sparsity_err = data_summary(dat.last, quo(sparse.error.max), 'sparsity error')
summary_dat_iteration = data_summary(dat.last %>% filter(solved.extrapolation == TRUE), quo(best.iteration), 'iteration')
summary_dat_solved_interp = data_summary(dat.last, quo(solved.interpolation), 'interpolation solved')
summary_dat_solved_extrap = data_summary(dat.last, quo(solved.extrapolation), 'extrapolation solved')

summary_dat = rbind(summary_dat_interp, summary_dat_extrap, summary_dat_sparsity_err, summary_dat_iteration, summary_dat_solved_interp, summary_dat_solved_extrap)
print(summary_dat)

write.csv(summary_dat, paste(results_folder, base_filename, '_', op, '_stats.csv', sep=''))

print("Script completed")
