rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
op <- args[4] # operation filter on (e.g. op-mul)
model_name <- args[5] # name of model to use in plot (i.e. short name). Use 'None' if you don't want to change the default model name. To be used on results file with only one model.
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
source('./_single_layer_task_expand_name.r')
source('../_compute_summary.r')
source('../_plot_parameter.r')

best.range = 5000

best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

first.solved.step = function (steps, errors, threshold) {
  index = first(which(errors <= threshold))
  if (is.na(index)) {
    return(NA)
  } else {
    return(steps[index])
  }
}

safe.interval = function (alpha, vec) {
  if (length(vec) <= 1) {
    return(NA)
  }

  return(abs(qt((1 - alpha) / 2, length(vec) - 1)) * (sd(vec) / sqrt(length(vec))))
}

name.parameter = 'interpolation.range'
name.label = 'Interpolation range'
name.file = paste(load_folder, base_filename, csv_ext, sep='') #'../results/function_task_static_mul_range.csv'
name.output = paste(results_folder, base_filename, '_', op, '.pdf', sep='') # '../paper/results/simple_function_static_mul_range.pdf'

csv.and.eps.merger = function (files.list, models.list, parameters.list) {
  # read in each file, rename the model to correct name, and concat all the tables row-wise

  load.eps = function(parameter.value) {
    # load the correct threshold value given the parameter
    eps = read_csv('./exp_setups.csv') %>%
          filter(simple == FALSE &  parameter == parameter.value & operation == op) %>%
          mutate(
            operation = revalue(operation, operation.full.to.short)
          ) %>%
          select(operation, extrapolation.range, epsilon) # READS EXTRAP COL WITH THE UNION THE UNICODE REF NOT THE SYMBOL
    return(eps)
  }

  merge.csvs = function(load.files.names, model.names) {
    combined.tables <- NULL
    # load tables for each element in the list AND EXPAND THEM
    tables <- lapply(lapply(load.files.names, read_csv), expand.name)
    for (idx in 1:length(tables)) {
      t <- ldply(tables[idx], data.frame)  # convert from list to df
      t$model <- model.names[[idx]]      # rename the model name to pre-defined value in list
      t <- inner_join(t, load.eps(parameters.list[idx]))  # merge the eps file info containing the extrap threshold
      combined.tables <- rbind(combined.tables, t)  # add model data to an accumulated table
    }
    return(combined.tables)
  }

  csvs.combined = merge.csvs(files.list, models.list)
  # dat needs model col to be a factor (because different models = different levels).
  # Without this line, you can't drop levels when plotting
  csvs.combined$model <- as.factor(as.vector(csvs.combined$model))
  return(csvs.combined)
}

load.csv.merge.eps = function(lookup.name) {
  return(switch(
    lookup.name,
    "nips-divBy0-easy" = csv.and.eps.merger(list(
      paste(load_folder, 'realnpu', csv_ext, sep = ''),
      paste(load_folder, 'nru', csv_ext, sep = ''),
      paste(load_folder, 'signNMRU', csv_ext, sep = '')
    ),
      list('Real NPU', 'NRU', 'NMRU'),
      list('zero.range.easy.realnpu', 'zero.range.easy', 'zero.range.easy.nmru')
    ),
    "nips-divBy0-medium" = csv.and.eps.merger(list(
      paste(load_folder, 'realnpu', csv_ext, sep = ''),
      paste(load_folder, 'nru', csv_ext, sep = ''),
      paste(load_folder, 'signNMRU', csv_ext, sep = '')
    ),
      list('Real NPU', 'NRU', 'NMRU'),
      list('zero.range.medium.realnpu', 'zero.range.medium', 'zero.range.medium.nmru')
    ),
    "nips-divBy0-hard" = csv.and.eps.merger(list(
      paste(load_folder, 'realnpu', csv_ext, sep = ''),
      paste(load_folder, 'nru', csv_ext, sep = ''),
      paste(load_folder, 'signNMRU', csv_ext, sep = '')
    ),
      list('Real NPU', 'NRU', 'NMRU'),
      list('zero.range.hard.realnpu', 'zero.range.hard', 'zero.range.hard.nmru')
    )
  ))
}


dat = load.csv.merge.eps(merge_mode)  %>%
  mutate(
    # !! = remember the expression I stored recently? Now take it, and ‘unquote’ it, that is, just run it!”
    parameter = !!as.name(name.parameter)
  )

dat.last = dat %>%
  group_by(name, parameter) %>%
  #filter(n() == 201) %>%
  summarise(
    threshold = last(epsilon),
    best.model.step = best.model.step.fn(metric.valid.interpolation),
    interpolation.last = metric.valid.interpolation[best.model.step],
    extrapolation.last = metric.test.extrapolation[best.model.step],
    interpolation.step.solved = first.solved.step(step, metric.valid.interpolation, threshold),
    extrapolation.step.solved = first.solved.step(step, metric.test.extrapolation, threshold),
    sparse.error.max = sparse.error.max[best.model.step],
    solved = replace_na(metric.test.extrapolation[best.model.step] <= threshold, FALSE),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, parameter) %>%
  group_modify(compute.summary)

if (model_name != 'None') {
  # Rename model with the given arg. The column is a factor, so the levels require renaming.
  levels(dat.last.rate$model) <- model_name
}

dat.last.rate$parameter <- gsub(']', ')', dat.last.rate$parameter)  # replace interp range notation from inclusion to exclusion i.e. ] to )

dat.gather = plot.parameter.make.data(dat.last.rate)

p = ggplot(dat.gather, aes(x = parameter, colour=model, group=interaction(parameter, model))) +
  geom_point(aes(y = mean.value), position=position_dodge(width=0.3)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value), position=position_dodge(width=0.3), alpha=0.5) +
  scale_color_discrete(limits = c('Real NPU', 'NRU', 'NMRU')) +
  scale_x_discrete(name = name.label) +
  scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
  scale_shape(guide = FALSE) +
  facet_wrap(~ key, scales='free_y', labeller = labeller(
    key = c(
      success.rate = "Success rate",
      converged.at = "Solved at iteration step",
      sparse.error = "Sparsity error"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) #+
  #geom_point(aes(y = max.value), position=position_dodge(width=0.3), shape=18) # plots max solved at step

ggsave(name.output, p, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
write.csv(dat.gather, paste(results_folder, base_filename, '_', op, '_plot_data.csv', sep=''))  # ADDED: save results table
write.csv(dat.last, paste(results_folder, base_filename, '_seeds_best', csv_ext, sep=''))            # best result for each seed
write_csv(filter(dat.last, solved == FALSE) %>% select(parameter, seed),  paste(results_folder, base_filename, '_seeds_failure', csv_ext , sep=''))

print("R Script completed.")
