rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)
source('./csv_merger.r')
source('../_function_task_expand_name.r')
source('../_compute_summary.r')
source('../_plot_parameter.r')

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results 
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
merge_mode <- args[4] # if 'None' then just loads single file. Otherwise looks up multiple results to merge together (use when have multiple models to plot)
merge_mode=ifelse(is.na(merge_mode), 'None', merge_mode)  # no passed arg becomes 'None' i.e. single model plot

csv_ext = '.csv'

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
name.output = paste(results_folder, base_filename, '.pdf', sep='') #'../paper/results/simple_function_static_mul_range.pdf'

#eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
eps = read_csv('exp_setups.csv') %>%
  filter(simple == FALSE & parameter == 'extrapolation.range') %>%
  mutate(
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, extrapolation.range, threshold)

# load (and merge) the exp results csvs and merge with the experument setup files
dat = load.and.merge.csvs(merge_mode)  %>%
  # to maintain ordering of dat use join not merge (otherwise the solved at subplot will be incorrect)
  inner_join(eps)  %>%
  mutate(
    # !! = remember the expression I stored recently? Now take it, and ‘unquote’ it, that is, just run it!”
    parameter = !!as.name(name.parameter)
  )

dat.last = dat %>%
  group_by(name, parameter, model) %>%
  #filter(n() == 201) %>%
  summarise(
    threshold = last(threshold),
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

dat.last.rate$parameter <- gsub(']', ')', dat.last.rate$parameter)  # replace interp range notation from inclusion to exclusion i.e. ] to )
dat.gather = plot.parameter.make.data(dat.last.rate)
# save accumulated df results to csv (both the ungrouped and grouped versions)
write.csv(dat.last.rate, paste0(results_folder, 'FTS_ranges_final_metrics', csv_ext)) 
write.csv(dat.gather, paste0(results_folder, 'FTS_ranges_gathered_metrics', csv_ext)) 


p = ggplot(dat.gather, aes(x = parameter, colour=model, group=interaction(parameter, model))) +
  geom_point(aes(y = mean.value), position=position_dodge(width=0.3)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value), position=position_dodge(width=0.3), alpha=0.5) +
  #scale_color_discrete("", limits=c('NMU', 'sNMU', 'PCC-MSE', 'bNAU')) + 
  scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
  scale_x_discrete(name = name.label) +
  scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
  scale_shape(guide = FALSE) +
  facet_wrap(~ key, scales='free_y', labeller = labeller(
    key = c(
      success.rate = "Extrapolation range success rate",
      converged.at = "Solved at iteration step",
      sparse.error = "Sparsity error"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#print(p)
ggsave(name.output, p, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")

write.csv(dat.last, paste0(results_folder, base_filename, '_seeds_best', csv_ext))            # best result for each seed
write.csv(dat.last.rate, paste0(results_folder, base_filename, '_final_metrics', csv_ext)) # Madsen eval metrics with confidence intervals 
write_csv(filter(dat.last, solved == FALSE) %>% select(name, parameter, seed),  paste0(results_folder, base_filename, '_seeds_failure', csv_ext))
print("R Script completed.")



