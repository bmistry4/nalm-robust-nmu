rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
model_name <- args[4] # name of model to use in plot (i.e. short name). Use 'None' if you don't want to change the default model name. To be used on results file with only one model.
model_name = ifelse(is.na(model_name), 'None', model_name)  # no passed arg becomes 'None' i.e. use default name
merge_mode <- args[5] # if 'None' then just loads single file. Otherwise looks up multiple results to merge together (use when have multiple models to plot)
merge_mode = ifelse(is.na(merge_mode), 'None', merge_mode)  # no passed arg becomes 'None' i.e. single model plot
csv_ext = '.csv'

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)
source('./csv_merger.r')

FILTER_EPOCH = -1  # set to the last epoch the model was trained on. Use -1 to get max value f.e. group.
name.parameter = 'epoch'  # column name containing x-axis values
name.label = 'Epoch'      # x-axis label
name.file = paste0(load_folder, base_filename, csv_ext)
name.output = paste0(results_folder, base_filename)

########################################################################################################################
### UTIL FUNCTIONS
# Get filter out each experiement to show the results for a specific step (i.e. epoch). -1 takes the max epoch per group (instead of a hardcoded filter epoch).
filter_by_step = function(dat, step_value = 100) {
  return(dat %>%
           group_by(name) %>%
           {if (step_value == -1) filter(.,epoch == max(epoch)) else .} %>% 
           {if (step_value != -1) filter(., epoch == step_value) else .} %>%
           ungroup()
  )
}


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

# plots models with test and train data using different colours
plot_scatter_with_se = function(plot_data, y_axis_label) {
  return(
    plot_data %>%
      ggplot(
        aes(x = model,
            y = mean,
            color = metric.type
        )) +
      scale_x_discrete(limits = c('mul', 'fc', 'nmu', 'snmu [1,5]', 'snmu [1,1+1/sd(x)]')) +
      geom_point(position = position_dodge(width = 0.3)) +
      geom_errorbar(aes(ymin = mean - se,
                        ymax = mean + se),
                    width = 0.3,
                    position = position_dodge(width = 0.3)
      ) +
      labs(y = y_axis_label, x = 'Model', color = 'Metric Type:') +
      theme(plot.margin = unit(c(5.5, 10.5, 5.5, 5.5), "points")) +
      #coord_flip() +
      theme(legend.position = "bottom")
  )
}

plot_acc_vs_ep = function(plot_data, y_axis_label) {
    plot_data$model <- factor(plot_data$model, levels = c('mul', 'fc', 'nmu', 'snmu [1,5]', 'snmu [1,1+1/sd(x)]'))
    return(
        plot_data %>%
          ggplot(
            aes(x = epoch,
                y = mean,
                color = model
            )) +
          geom_line(aes(fill=model)) +
          geom_ribbon(aes(ymin = mean-se, ymax = mean+se, fill=model), alpha = 0.05, size=0.01, color='grey') +
          #scale_x_discrete(limits = rev) + # reverse order of categorical variables
          labs(y = y_axis_label, x = 'Epoch') +
          theme(plot.margin = unit(c(5.5, 10.5, 5.5, 5.5), "points")) +
          theme(legend.position = "bottom")
  )
}

########################################################################################################################
### PLOTTING MSE WITH ERROR BARS
# load the data
expanded_dat = load_and_merge_csvs(merge_mode, single_filepath = paste0(load_folder, base_filename, '.csv'))
# filter to single epoch so each fold for each model has 1 result (row of data)
expanded_dat = filter_by_step(expanded_dat, step_value = FILTER_EPOCH)
expanded_dat = expanded_dat %>% group_by(model)

# Rename model with the given arg (use for single file processing)
if (model_name != 'None') {
  expanded_dat$model = model_name
}

# summarise the test and train data
summary_dat_test_output = data_summary(expanded_dat, quo(metric.test.output.mse), 'output (test)')
summary_dat_train_output = data_summary(expanded_dat, quo(metric.train.output.mse), 'output (train)')

summary_dat_test_lab1 = data_summary(expanded_dat, quo(metric.test.label1.mse), 'label 1 (test)')
summary_dat_train_lab1 = data_summary(expanded_dat, quo(metric.train.label1.mse), 'label 1 (train)')

summary_dat_test_lab2 = data_summary(expanded_dat, quo(metric.test.label2.mse), 'label 2 (test)')
summary_dat_train_lab2 = data_summary(expanded_dat, quo(metric.train.label2.mse), 'label 2 (train)')

# plot the summary stats for each model
summary_dat = rbind(summary_dat_train_output, summary_dat_test_output,
                    summary_dat_train_lab1, summary_dat_test_lab1,
                    summary_dat_train_lab2, summary_dat_test_lab2)
print(summary_dat)
ci_plot = plot_scatter_with_se(summary_dat, y_axis_label = 'MSE')
#ci_plot

# save pdf and the the data to generate the pdf.
ggsave(paste0(name.output, '_mse', '.pdf'), ci_plot, device = "pdf")
write.csv(summary_dat, paste0(name.output, '_mse', '_plot_data.csv'))

########################################################################################################################
### PLOTTING ACCURACY WITH ERROR BARS
# load the data
expanded_dat = load_and_merge_csvs(merge_mode, single_filepath = paste0(load_folder, base_filename, '.csv'))
# filter to single epoch so each fold for each model has 1 result (row of data)
expanded_dat = filter_by_step(expanded_dat, step_value = FILTER_EPOCH)
expanded_dat = expanded_dat %>% group_by(model)

# Rename model with the given arg (use for single file processing)
if (model_name != 'None') {
  expanded_dat$model = model_name
}

# summarise the test and train data
summary_dat_test_output = data_summary(expanded_dat, quo(metric.test.output.acc), 'output (test)')
summary_dat_train_output = data_summary(expanded_dat, quo(metric.train.output.acc), 'output (train)')

summary_dat_test_lab1 = data_summary(expanded_dat, quo(metric.test.label1.acc), 'label 1 (test)')
summary_dat_train_lab1 = data_summary(expanded_dat, quo(metric.train.label1.acc), 'label 1 (train)')

summary_dat_test_lab2 = data_summary(expanded_dat, quo(metric.test.label2.acc), 'label 2 (test)')
summary_dat_train_lab2 = data_summary(expanded_dat, quo(metric.train.label2.acc), 'label 2 (train)')

# plot the summary stats for each model
summary_dat = rbind(summary_dat_train_output, summary_dat_test_output,
                    summary_dat_train_lab1, summary_dat_test_lab1,
                    summary_dat_train_lab2, summary_dat_test_lab2)
print(summary_dat)
ci_plot = plot_scatter_with_se(summary_dat, y_axis_label = 'Accuracy (%)')
#ci_plot

# save pdf and the the data to generate the pdf.
ggsave(paste0(name.output, '_acc', '.pdf'), ci_plot, device = "pdf")
write.csv(summary_dat, paste0(name.output, '_acc', '_plot_data.csv'))
########################################################################################################################
### PLOTTING ACCURACY VS EPOCHS # TODO cleanup and make it for each
# load the data
expanded_dat = load_and_merge_csvs(merge_mode, single_filepath = paste0(load_folder, base_filename, '.csv'))
expanded_dat = expanded_dat %>% group_by(model, epoch)

# Rename model with the given arg (use for single file processing)
if (model_name != 'None') {
  expanded_dat$model = model_name
}

# summarise the test and train data
summary_dat_test_output = data_summary(expanded_dat, quo(metric.test.output.acc), 'output (test)')
summary_dat_test_lab1 = data_summary(expanded_dat, quo(metric.test.label1.acc), 'label 1 (test)')
summary_dat_test_lab2 = data_summary(expanded_dat, quo(metric.test.label2.acc), 'label 2 (test)')

# plot the summary stats for each model
epoch_plot = plot_acc_vs_ep(summary_dat_test_output, 'Test Accuracy (%)')
# save pdf and the the data to generate the pdf.
#ggsave(paste0(name.output, '_ep-acc_NO-RIBBON', '.pdf'), epoch_plot, device = "pdf")
ggsave(paste0(name.output, '_ep-acc', '.pdf'), epoch_plot, device = "pdf")
write.csv(summary_dat_test_output, paste0(name.output, '_ep-acc', '_plot_data.csv'))

# plot the summary stats for each model
epoch_plot = plot_acc_vs_ep(summary_dat_test_lab1, 'Left Label Accuracy (%)')
# save pdf and the the data to generate the pdf.
ggsave(paste0(name.output, '_ep-acc-lab1', '.pdf'), epoch_plot, device = "pdf")
write.csv(summary_dat_test_lab1, paste0(name.output, '_ep-acc-lab1', '_plot_data.csv'))

# plot the summary stats for each model
epoch_plot = plot_acc_vs_ep(summary_dat_test_lab2, 'Right Label Accuracy (%)')
# save pdf and the the data to generate the pdf.
ggsave(paste0(name.output, '_ep-acc-lab2', '.pdf'), epoch_plot, device = "pdf")
write.csv(summary_dat_test_lab2, paste0(name.output, '_ep-acc-lab2', '_plot_data.csv'))

########################################################################################################################

print("R Script completed.")
