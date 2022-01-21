# Plot the ratio between the distance travelled between weight:gate for the given epoch interval
# NOTE: struggle with 0 distance cases
rm(list = ls())
setwd('C:/Users/mistr/Documents/SOTON/PhD/Code/nalu-stable-exp/export/single_layer_task/')

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
ending <- args[4] # name to add to end of saved file (before the extension)
plot_seed <- as.integer(args[5]) # which seed to plot. If empty/NA will plot all seeds.
plot_l2 <- args[6]  # plot l2 dist (TRUE) or the weight path (FALSE)
load_file = paste(load_folder, base_filename, '.csv', sep = '')
results_file = paste(results_folder, base_filename, "_s", plot_seed, ending, '.pdf', sep = '')

load_file = 'log-10_weightsPath_realnpu-mod_i--2-2_successes.csv' # 'log-10_weightsPath_realnpu-mod_i--2-2_s20.csv' # 'log-10_weightsPath_realnpu-mod_i--2-2_successes.csv' #
plot_seed = NA
plot_l2 = TRUE

library(ggplot2)
library(dplyr)
library(readr)
source('./_single_layer_task_expand_name.r')

dat <- read_csv(load_file)
dat <- filter(dat, step %% 10 == 0) # reduce the amount of data to plot
eps <- 0.01 #.Machine$double.eps

# parse seed value
dat <- dat %>%
  rowwise() %>%
  mutate(seed = as.integer(substring(extract.by.split(name, 12), 2)))

if (!is.na(plot_seed)) {
  dat <- filter(dat, seed == plot_seed)
}

# plot the l2 distances
if (plot_l2) {
  dat$l2.weight <- NA
  distances.l2.weight <- sqrt(diff(dat$weights.w0)^2 + diff(dat$weights.w1)^2)
  dat$l2.weight[-1] <- distances.l2.weight  # offset so first epoch doesn't have a distance
  dat <- dat %>% mutate(l2.weight = ifelse(step == 0, NA, l2.weight)) # epoch 0 f.e. seed will have no point plotted

  dat$l2.gate <- NA
  distances.l2.gate <- sqrt(diff(dat$gate.g0)^2 + diff(dat$gate.g1)^2)
  dat$l2.gate[-1] <- distances.l2.gate  # offset so first epoch doesn't have a distance
  dat <- dat %>% mutate(l2.gate = ifelse(step == 0, NA, l2.gate)) # epoch 0 f.e. seed will have no point plotted

  dat <- filter(dat, step %% 100 == 0) # filter now so the l2 dists aren't calculated over a long interval

  #dat <- dat %>%
  #  mutate(
  #    l2.weight = ifelse(l2.weight == 0, 1, l2.weight),
  #    l2.gate = ifelse(l2.gate == 0, 1, l2.gate),
  #  )

  # add eps to deal with /0
  dat$l2.ratio.w2g <- (dat$l2.weight + eps) / (dat$l2.gate + eps)


  p =
    ggplot(dat, aes(x = step, y = l2.ratio.w2g, color = as.factor(seed), group = seed)) +
      #coord_cartesian(xlim = c(-1, 1), ylim = c(-1, 1)) +
      geom_point(size = 0.2) +
      geom_path(size = 0.01) +
      geom_line(y=1, color='black') +
      xlab("Epoch") +
      #ylab("L2 distance") +
      ylab("weight:gate L2 ratio") +
      theme(legend.position = "bottom", plot.margin = unit(c(1, 1, 1, 1), "cm")) +
      scale_colour_discrete("Seed") #+
      #coord_cartesian(xlim = c(0, 50000), ylim = c(0, 4))


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

ggsave('w2g-ratio_mse_success.pdf', p, device = "pdf", width = 4, height = 4, scale = 4, units = "cm")
#ggsave(results_file, p, device = "pdf", width = 4, height = 4, scale = 4, units = "cm")
