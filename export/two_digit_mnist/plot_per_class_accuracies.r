# plots the accuracy f.e. digit class from a pretrained model's confusion matrix. Will plot zoomed in version on left and full plot on right.

rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

# TODO change to args
load_folder = "C:\\Users\\mistr\\Documents\\SOTON\\PhD\\Code\\nalu-stable-exp\\save\\two_digit_mnist_plots\\"
results_folder = "C:\\Users\\mistr\\Documents\\SOTON\\PhD\\Code\\nalu-stable-exp\\save\\two_digit_mnist_plots\\"
merge_mode = 'mul-1digit_conv-Adam'

library(ggplot2)
library(ggforce)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)
library(ggbreak)

csv_merger = function(load_files_names, models_name_list) {
  combined_tables <- NULL
  # load tables for each element in the list
  tables <- lapply(load_files_names, read_csv)
  for (idx in 1:length(tables)) {
    t <- ldply(tables[idx], data.frame)  # convert from list to df
    # don't process dfs with no rows - to avoid dists where all configs failed to reach required max step
    if (!empty(t)) {
      # rename model if names have been given
      if (length(models_name_list)) {
        t$model <- models_name_list[[idx]]      # rename the model name to pre-defined value in list
      }
      combined_tables <- rbind(combined_tables, t)  # add model data to an accumulated table
    }
  }
  return(combined_tables)
}

load_and_merge_csvs = function(lookup.name) {
  csv_ext = '.csv'
  return(switch(
    lookup.name,
    "mul-1digit_conv-Adam" = csv_merger(list(
      paste0(load_folder, '23_f2_op-mul_nalmF_learnLF_s2_class_accs', csv_ext),
      paste0(load_folder, '25_f2_op-mul_nalmF_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '24_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '22_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '70_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'snmu', 'snmu [1,1+1/sd(x)')
    ),
    "MSE_Adam-lr0.001_TPS-no-concat-conv_ROUNDED" = csv_merger(list(
      paste0(load_folder, '60_f2_op-mul_nalmF_learnLF_s2_class_accs', csv_ext),
      paste0(load_folder, '63_f2_op-mul_nalmF_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '62_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '61_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '69_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'snmu', 'snmu [1,1+1/sd(x)')
    ),
    "MSE_Adam-lr0.001_TPS-no-concat-conv_NOT-ROUNDED" = csv_merger(list(
      paste0(load_folder, '60_f2_op-mul_nalmF_learnLF_s2_class_accs', csv_ext),
      paste0(load_folder, '63_f2_op-mul_nalmF_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '62_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '61_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext),
      paste0(load_folder, '69_f2_op-mul_nalmT_learnLT_s2_class_accs', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'snmu', 'snmu [1,1+1/sd(x)')
    )
  ))
}

expanded_dat = load_and_merge_csvs(merge_mode)
expanded_dat = expanded_dat %>% group_by(model)

expanded_dat$model <- factor(expanded_dat$model, levels = c('mul', 'fc', 'nmu', 'snmu', 'snmu [1,1+1/sd(x)'))
plot = ggplot(expanded_dat, aes(x = factor(digit), y = accuracy, fill = model,
                                width=0.6)) +
        labs(y = 'Success rate', x = 'Digit') +
        geom_bar(stat="identity", position = "dodge") +
        facet_zoom(ylim = c(0.96, 1)) +
        theme(legend.position = "bottom") #+
        #scale_y_cut(breaks=c(0.1, 0.78), , which=c(2,3), scales=0.1)
#plot
ggsave(paste0(results_folder, merge_mode, '_class_accs.pdf'), plot, , width = 13.968, height = 5.7, scale=2, units = "cm", device = "pdf")
write.csv(expanded_dat, paste0(results_folder, merge_mode, '_combined_class_accs.csv'))  # ADDED: save results table

