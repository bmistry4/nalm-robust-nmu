# Sor a module with 2 weights, plot the path of the weights which are coloured depending on the epoch value
rm(list = ls())
#setwd('C:/Users/mistr/Documents/SOTON/PhD/Code/nalu-stable-exp/export/single_layer_task/')

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
ending <- args[4] # name to add to end of saved file (before the extension)
plot_seed <- as.integer(args[5]) # which seed to plot. If empty/NA will plot all seeds.
plot_l2 <- args[6]  # plot l2 dist (TRUE) or the weight path (FALSE)
load_file = paste(load_folder, base_filename, '.csv', sep = '')
results_file = paste(results_folder, base_filename, "_s", plot_seed, ending, '.pdf', sep = '')

library(ggplot2)
library(dplyr)
library(readr)
source('./_single_layer_task_expand_name.r')

dat <- read_csv(load_file)
dat <- filter(dat, step %% 1000 == 0) # reduce the amount of data to plot

# parse seed value
dat <- dat %>%
  rowwise() %>%
  mutate(seed = as.integer(substring(extract.by.split(name, 12), 2)))

if (!is.na(plot_seed)) {
  dat <- filter(dat, seed == plot_seed)
}

# plot the l2 distances
if (plot_l2) {
  dat$l2 <- NA
  distances.l2 <- sqrt(diff(dat$weights.w0)^2 + diff(dat$weights.w1)^2)
  dat$l2[-1] <- distances.l2  # offset so first epoch doesn't have a distance
  dat <- dat %>% mutate(l2 = ifelse(step == 0, NA, l2)) # epoch 0 f.e. seed will have no point plotted

  p =
    ggplot(dat, aes(x = step, y = l2, color = as.factor(seed), group = seed)) +
      #coord_cartesian(xlim = c(-1, 1), ylim = c(-1, 1)) +
      geom_point(size = 0.2) +
      geom_path(size = 0.01) +
      xlab("Epoch") +
      ylab("L2 distance") +
      theme(legend.position = "bottom", plot.margin = unit(c(1, 1, 1, 1), "cm")) +
      scale_colour_discrete("Seed")

} else {
  # plot the weight paths

  # get the first and last logged data rows (per seed group)
  data.first.epoch <- dat %>% group_by(seed) %>% slice_head(n = 1)
  data.last.epoch <- dat %>% group_by(seed) %>% slice_tail(n = 1)

  p =
    ggplot(dat, aes(x = weights.w0, y = weights.w1, color = step, group = seed)) +
      coord_cartesian(xlim = c(-1, 1), ylim = c(-1, 1)) +
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
      theme(plot.margin = unit(c(1, 1, 1, 1), "cm")) +
      geom_point(aes(x = gate.g0, y = gate.g1, color = step), size = 0.2) +
      geom_path(aes(x = gate.g0, y = gate.g1, color = step), size = 0.01) +
      geom_point(aes(x = gate.g0, y = gate.g1, color = step),
                 data = data.first.epoch, color = "blue", size = 1, shape = 2) +
      geom_point(aes(x = gate.g0, y = gate.g1, color = step),
                 data = data.last.epoch, color = "red", size = 1, shape = 2)
}

ggsave(results_file, p, device = "pdf", width = 4, height = 4, scale = 4, units = "cm")
