# only plots the success rate

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
model_names_list <- args[8]  # the order for the categorical variables to appear. Example of passing args: "Real NPU (baseline), Real NPU (modified), NRU, NMRU"
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
source('./npu_csv_merger.r')

# prase command line args for x axis model names. 
#model_names_list = strsplit(model_names_list, ", ")[[1]]

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
  index = first(which(errors < threshold))
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

# single layer task version of funcation_task_static_mse_expectation.csv
eps = read_csv('./exp_setups.csv') %>% 
  filter(simple == FALSE & parameter == parameter_value & operation == op) %>%
  mutate(
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, extrapolation.range, epsilon) # THIS IS THE PROBLEM! READS THE EXTRAP COL WITH THE UNION THE UNICODE REF NOT THE SYMBOL


dat = load.and.merge.csvs(merge_mode)  %>%
  # to maintain ordering of dat use join not merge (otherwise the solved at subplot will be incorrect)
  inner_join(eps)  %>%
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
    solved = replace_na(metric.test.extrapolation[best.model.step] < threshold, FALSE),
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
dat.gather = dat.gather %>% filter(key == 'success.rate')  # filter to show success rate metrics only 

p = ggplot(dat.gather, aes(x = parameter, colour=model, group=interaction(parameter, model))) +
  geom_point(aes(y = mean.value), position=position_dodge(width=0.3)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value), position=position_dodge(width=0.3), alpha=0.5) +
  # for a custom label order replace labels = model.to.exp(levels(dat.gather$model)) with limits=c(<"model1">, <"model2">, <"model3">) (copied from npu_csv_merger.r)
  #scale_color_discrete("", limits=c('None', 'G', 'W', 'GW')) +         # legend title and ordering 
  #scale_color_discrete("", limits=c(model_names_list)) +               # legend title and ordering when model names order is given through cmmd line arg
  scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) + 
  scale_x_discrete(name = name.label) +
  scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
  scale_shape(guide = FALSE) +
  facet_wrap(~ key, scales='free_y', labeller = labeller(
    key = c(
      success.rate = "Extrapolation range success rate"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) #+
  #guides(col = guide_legend(nrow = 2, byrow = TRUE))      # wrap legend around n rows

ggsave(name.output, p, device="pdf", width = 5, height = 5.7, scale=1.4, units = "cm")
write.csv(dat.gather, paste(results_folder, base_filename, '_', op, '_plot_data.csv', sep=''))  # ADDED: save results table 
#write.csv(dat.last, paste(results_folder, base_filename, '_seeds_best', csv_ext, sep=''))            # best result for each seed
#write_csv(filter(dat.last, solved == FALSE) %>% select(parameter, seed),  paste(results_folder, base_filename, '_seeds_failure', csv_ext , sep=''))

print("R Script completed.")
