# Run this file to generate the gold extrapolation error over different ranges and 3 different division toy tasks.
rm(list = ls())
#setwd('C:/Users/mistr/Documents/SOTON/PhD/Code/nalu-stable-exp/export/single_layer_task/')

library(ggplot2)
library(dplyr)
library(readr)

load_file = "divBy0_test_errors.csv"

dat = read_csv(load_file) %>%
  # rename to realnpu and nru
  #filter(parameter == 'zero.range.easy.realnpu'
  #        | parameter == 'zero.range.medium.realnpu'
  #        | parameter == 'zero.range.hard.realnpu'
  #        | parameter == 'zero.range.easy'
  #        | parameter == 'zero.range.medium'
  #        | parameter == 'zero.range.hard'
  #) %>%
  mutate(
  # have key for parameter: easy, med, hard
  #key = factor(parameter, levels = c("zero.range.easy", "zero.range.medium", "zero.range.hard"))
  key = factor(parameter, levels = c("easy", "medium", "hard"))
  )

# Convert cyl as a grouping variable
dat$parameter <- as.factor(dat$parameter)
#dat$range <- gsub(']', ')', dat$range)  # replace interp range notation from inclusion to exclusion i.e. ] to )

# manually order x axis. If this is used, then comment out scale_x_discrete(limits=rev)
#dat$extrapolation.range <- factor(dat$extrapolation.range,levels = c("U[0,1]", "U[0,0.1]", "U[0,0.01]", "U[0,0.001]", "U[0,0.0001]"))
#dat$range <- factor(dat$range,levels = c("[0, 1]", "[0, 0.1]", "[0, 0.01]", "[0, 0.001]", "[0, 0.0001]", "[0, 1e-05]"))
dat$range <- factor(dat$range,levels = c('U[0,1e-8)', 'U[0,1e-7)', 'U[0,1e-6)', 'U[0,1e-5)', 'U[0,1e-4)', 'U[0,1e-3)', 'U[0,1e-2)', 'U[0,1e-1)', 'U[0,1e+0)'))

expandy = function(vec, ymin=NULL) {
  max.val = max(vec, na.rm=TRUE)
  min.val = min(vec, na.rm=TRUE)
  min.log = floor(log10(max.val))
  expand_limits(y=c(min.val, ceiling(max.val/10^min.log)*10^min.log))
}


p =
  ggplot(dat, aes(x=range, y=test.mse, color=module)) +
    scale_y_log10() +                           # log scaling
    geom_point() +
    #scale_x_discrete(limits=rev)+               # x axis in reverse order
    geom_line(aes(group=module)) +
    xlab("Data sample range") +
    ylab("log(Test error)") +
    theme(legend.position="bottom") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    #coord_cartesian( ylim = c(1e-7, 1e+13)) +  # set y axis bounds
    facet_wrap(~ key, scales='free_y', labeller = labeller(
      key = c(
        easy = "Input: [a]; Output: 1/a",
        medium =  "Input: [a,b]; Output: 1/a",
        hard =  "Input: [a,b]; Output: a/b"
      )
    )) +
    scale_color_discrete("", limits=c('RealNPU', 'RealNPU (eps=0)', 'NRU', 'NMRU')) +
    theme(panel.spacing.x=unit(1, "lines")) +
    coord_cartesian(ylim = c(1e-13, 1e+20)) #+     # Set y axis limits manually
    #geom_blank(data = data.frame(
    #  key = c("easy"),
    #  model = NA,
    #  y.limit.max = c(8000),
    #  y.limit.min = c(1e-12),
    #  parameter = dat$test.mse
    #), aes(x = range, y = y.limit.max))
    #scale_fill_discrete(name = "Title",  labels = c("A", "B"))
    #theme(plot.margin=unit(c(1,1,1,1), "cm"))  # padding on edges of plot

ggsave("divBy0_gold_test_errors.pdf", p, device = "pdf", width = 13.968, height = 5.7, scale = 1.4, units = "cm")
