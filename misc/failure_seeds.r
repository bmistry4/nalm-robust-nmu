# saves csv of seeds which failed. Requires a seed_best.csv to have been created to work (which can be created by calling range.r with the save line uncommented).
# usage: Rscript failure_seeds.r /data/bm4g15/nalu-stable-exp/csvs/r_results/single_layer_task/NRU/SLTR_NRU-tanh1000_lr-1e-3_inSize-10_stable-pcc_E100k_Reg-50k-75k_seeds_best
library(readr)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
base_filename <- args[1] # filepath with the _seeds_best.csv removed'
load_name_ending <- '_seeds_best.csv'
save_name_ending <- '_seeds_failure.csv'  
load_file <- paste(base_filename, load_name_ending, sep='')
save_file <- paste(base_filename, save_name_ending, sep='')

dat <- read_csv(load_file) 
fail_seeds <- filter(dat, solved == FALSE) %>% select(parameter, seed)
print(fail_seeds)
write_csv(fail_seeds, save_file)


