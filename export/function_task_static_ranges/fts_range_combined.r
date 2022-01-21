rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)
source('../_function_task_expand_name.r')
source('../_compute_summary.r')
source('../_plot_parameter.r')

# commdand: Rscript fts_range_combined.r /data/bm4g15/nalu-stable-exp/csvs/r_results/FTS_NAU_NMU_ranges/ /data/bm4g15/nalu-stable-exp/csvs/r_results/FTS_NAU_NMU_ranges/FINAL/

args <- commandArgs(trailingOnly = TRUE) 
results_folder <- args[1] # folder to load the final_metrics csvs
save_folder <- args[2] # folder to save the plot
save_filename = 'FTS_ranges_combined'
csv_ext = '.csv'
name.output = paste(save_folder, save_filename, '.pdf', sep='') #output plot filepath

name.parameter = 'interpolation.range'
name.label = 'Interpolation range'

# list of filepaths to the final_metrics results for each different model. (These results will be combined onto the same plot)
files.list = list(
  paste(results_folder, 'FTS_NAU_NMU_ranges_baseline', '_final_metrics', csv_ext, sep=''),
  paste(results_folder, 'FTS_NAU_NMU_ranges_noise-1-5', '_final_metrics', csv_ext, sep=''),
  paste(results_folder, 'FTS_nau-Nnmu-epsPcc-750K-mse', '_final_metrics', csv_ext, sep=''),
  paste(results_folder, 'FTS_beta-nau-Nnmu_ranges-pcc-750K-mse', '_final_metrics', csv_ext, sep='')
)

# Name of the models to associate with each file in the above list. Currently they will all be named NMU.
models.list = list('NMU', 'sNMU', 'PCC-MSE', 'bNAU')

# read in each file, rename the model to correct name, and concat all the tables row-wise
merge.final.metrics.csvs = function (load.files.names, model.names) {
  combined.tables = NULL
  tables <- lapply(load.files.names, read_csv)  # load tables for each element in the list
  for (idx in 1:length(tables)) {
    t <- ldply(tables[idx], data.frame)  # convert from list to df
    t$model <- model.names[[idx]]      # rename the model name to pre-defined value in list 
    combined.tables = rbind(combined.tables, t)  # add model data to an accumulated table 
  }  
  return(combined.tables)
}

dat.last.rate = merge.final.metrics.csvs(files.list, models.list)%>%
  group_by(model, operation, parameter)

#name.file = 'final_metrics_test.csv'
#dat.last.rate = read_csv(name.file)%>%
#  group_by(model, operation, parameter)  

post.plot.parameter.make.data = function (dat.last.rate, ...) {
  dat.gather.mean = dat.last.rate %>%
    mutate(
      success.rate = success.rate.mean,
      converged.at = converged.at.mean,
      sparse.error = sparse.error.mean
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
    gather('key', 'mean.value', success.rate, converged.at, sparse.error)
  
  dat.gather.upper = dat.last.rate %>%
    mutate(
      success.rate = success.rate.upper,
      converged.at = converged.at.upper,
      sparse.error = sparse.error.upper
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
    gather('key', 'upper.value', success.rate, converged.at, sparse.error)
  
  dat.gather.lower = dat.last.rate %>%
    mutate(
      success.rate = success.rate.lower,
      converged.at = converged.at.lower,
      sparse.error = sparse.error.lower
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
    gather('key', 'lower.value', success.rate, converged.at, sparse.error)
  
  dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
    mutate(
      #model=droplevels(model),  # FIXME: DOESN'T LIKE THIS LINE 
      #our.model=model %in% c('NMU', 'NAU'),  # DON'T NEED THIS LINE
      key = factor(key, levels = c("success.rate", "converged.at", "sparse.error")),
    )
  
  return(dat.gather)
}

dat.last.rate$parameter <- gsub(']', ')', dat.last.rate$parameter)  # replace interp range notation from inclusion to exclusion i.e. ] to )
dat.gather = post.plot.parameter.make.data(dat.last.rate)
# save accumulated df results to csv (both the ungrouped and grouped versions)
write.csv(dat.last.rate, paste(save_folder, 'FTS_ranges_final_metrics', csv_ext, sep='')) 
write.csv(dat.gather, paste(save_folder, 'FTS_ranges_gathered_metrics', csv_ext, sep='')) 

print(dat.last.rate)
p = ggplot(dat.gather, aes(x = parameter, colour=model, group=interaction(parameter, model))) +
  geom_point(aes(y = mean.value), position=position_dodge(width=0.3)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value), position=position_dodge(width=0.3), alpha=0.5) +
  #scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) + 
  scale_color_discrete(labels = levels(factor(dat.gather$model))) +  # FIX to make model names show in key
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
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(name.output, p, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
print("R Script completed.")
