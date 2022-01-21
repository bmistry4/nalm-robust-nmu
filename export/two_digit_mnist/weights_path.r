# For a module with 2 weights, plot the path of the weights which are coloured depending on the epoch value
rm(list = ls())

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
ending <- args[4] # name to add to end of saved file (before the extension)
plot_fold <- as.integer(args[5]) # which seed to plot. If empty/NA will plot all seeds.

load_file = paste0(load_folder, base_filename, '.csv')
results_file = paste0(results_folder, base_filename, "_f", plot_fold, ending, '.pdf')

library(ggplot2)
library(plyr)
library(dplyr)
library(readr)
source('./csv_merger.r')


dat <- load_and_merge_csvs('None', single_filepath = paste0(load_folder, base_filename, '.csv'))
dat <- filter(dat, step %% 1 == 0) # reduce the amount of data to plot

if (!is.na(plot_fold)) {
  dat <- filter(dat, fold == plot_fold)
}

data.first.epoch <- dat %>% group_by(fold) %>% slice_head(n = 1)
data.last.epoch <- dat %>% group_by(fold) %>% slice_tail(n = 1)

p = ggplot(dat, aes(x = label2out.w0, y = label2out.w1, color = epoch, group = fold)) +
    coord_cartesian(xlim = c(0, 1.5), ylim = c(0, 1.5)) +
    geom_point(size = 0.2) +
    geom_path(size = 0.01) +
    geom_point(data = data.first.epoch, color = "blue", size = 1, shape = 8) +
    geom_point(data = data.last.epoch, color = "red", size = 1, shape = 8) +
    #geom_text(data = data.first.epoch, label = "start", vjust = -1.5, color = "black") +
    #geom_text(data = data.last.epoch, label = "end", vjust = -1.5, color = "black") +
    xlab("Weight 0") +
    ylab("Weight 1") +
    theme(legend.position = "bottom", legend.text = element_text(angle = 45, hjust = 1)) +
    scale_color_gradient(low = "blue", high = "red", name = "epoch") +
    theme(plot.margin = unit(c(1, 1, 1, 1), "cm")) #+

#p
ggsave(results_file, p, device = "pdf", width = 4, height = 4, scale = 4, units = "cm")
write.csv(dat, paste0(results_file, '_plot_data.csv'))

