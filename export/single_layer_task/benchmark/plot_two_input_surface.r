rm(list = ls())

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
merge_mode <- args[4] # if 'None' then just loads single file. Otherwise looks up multiple results to merge together (use when have multiple models to plot)
merge_mode = ifelse(is.na(merge_mode), 'None', merge_mode)  # no passed arg becomes 'None' i.e. single model plot
num_bins <- args[5]
num_bins = ifelse(is.na(num_bins), 20, as.numeric(num_bins)) 

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)

print(load_folder)
print(results_folder)
print(base_filename)
print(merge_mode)
print(num_bins)

interp_range_low = 1
interp_range_high = 2
extrap_range_low = 2
extrap_range_high = 6

csv_ext = '.csv'
name.output = paste0(results_folder, base_filename)

csv.merger = function(files.list, models.list) {
  # read in each file, rename the model to correct name, and concat all the tables row-wise
  merge.csvs = function(load.files.names, model.names, interp.solved, extrap.solved) {
    combined.tables <- NULL
    # load tables for each element in the list
    tables <- lapply(load.files.names, read_csv)
    for (idx in 1:length(tables)) {
      t <- ldply(tables[idx], data.frame)  # convert from list to df
      t$model <- model.names[[idx]]      # rename the model name to pre-defined value in list
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

load.and.merge.csvs = function(lookup_name, load_folder, csv_ext='.csv') {
  return(switch(
    lookup_name,
    "mul" = csv.merger(list(
      paste0(load_folder, 'mul_gold', csv_ext),
      paste0(load_folder, 'mul-nmu-s0', csv_ext),
      paste0(load_folder, 'mul-ReLU-h1-s0', csv_ext),
      paste0(load_folder, 'mul-ReLU-h100-s8', csv_ext)
    ),
      list('Gold (TT)', 'NMU (TT)', 'MLP (1) (FF)', 'MLP (100) (TF)')
    ),
    "add" = csv.merger(list(
      paste0(load_folder, 'add_gold', csv_ext),
      paste0(load_folder, 'add-nau-s0', csv_ext),
      paste0(load_folder, 'add-ReLU-h1-s9', csv_ext),
      paste0(load_folder, 'add-ReLU-h1-s12', csv_ext),
      paste0(load_folder, 'add-ReLU-h4-biasF-s4', csv_ext),
      paste0(load_folder, 'add-ReLU-h4-biasF-s0', csv_ext),
      paste0(load_folder, 'add-ReLU-h4-biasF-s17', csv_ext),
      paste0(load_folder, 'add-ReLU-h100-s0', csv_ext)
    ),
      list('Gold (TT)', 'NAU (TT)', 
            'MLP (1) (TT)', 'MLP (1) (FF)',
            'MLP (4) (TT)', 'MLP (4) (TF)', 'MLP (4) (FF)',
            'MLP (100) (TF)'
      )
    )
  ))
}

dat = load.and.merge.csvs(merge_mode, load_folder) %>% group_by(model)
print('csvs loaded and merged')

# Create ten segments to be colored in
dat$equalSpace <- cut(dat$pred, num_bins)
breaks <- levels(unique(dat$equalSpace))
# 10 bin colours -> c("#35978f", "#80cdc1", "#c7eae5", "#f5f5f5","#f6e8c3", "#dfc27d", "#bf812d", "#8c510a", "#543005", "#330000")
bin_colours <- colorRampPalette(c("#35978f", "#c7eae5", "#bf812d", "#330000"))(num_bins)

# Plot
p <- ggplot(dat, aes(x1, x2)) +
  geom_tile(aes(fill = equalSpace)) +
  theme_bw() +
  guides(fill=guide_legend(nrow=num_bins/5,byrow=TRUE)) +
  theme(legend.position="bottom", legend.key.size = unit(0.3, 'cm')) +
  xlab("x1") +
  ylab("x2") +
  geom_rect(mapping = aes(xmin = interp_range_low, xmax = interp_range_high, ymin = interp_range_low, ymax = interp_range_high), 
    color = "blue", alpha = 0.0) +  # draw box around interp range
  geom_rect(mapping = aes(xmin = extrap_range_low, xmax = extrap_range_high, ymin = extrap_range_low, ymax = extrap_range_high), 
    color = "red", alpha = 0.0) +  # draw box around extrapo range 
  scale_fill_manual(values=bin_colours, name = "") +              # use name to give legend a title
  #facet_wrap(~factor(model, levels=c('Gold (TT)', 'NMU (TT)', 'MLP (1) (FF)', 'MLP (100) (TF)')), nrow=2)  # use for mul                  
  facet_wrap(~factor(model, 
              levels=c('Gold (TT)', 'NAU (TT)', 
                        'MLP (1) (TT)', 'MLP (1) (FF)',
                        'MLP (4) (TT)', 'MLP (4) (TF)', 'MLP (4) (FF)',
                        'MLP (100) (TF)')
             ), 
              nrow=2)   # use for add 

print('plot created')
ggsave(paste0(name.output, '.pdf'), p, device = "pdf", width = 16, height = 10, scale = 1.4, units = "cm")
print('plot saved')
