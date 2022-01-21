rm(list = ls())

library(readr)

# This script will take csv files with the same headings, stack them row-wise and save the resulting dataset.
# Used to combine the single layer task range results with the later added units e.g. NALU sub units

# Used to filter out the single layer task range results for add and sub and combine the new NAU sub and add results. And will combine the NALU sub units results.
file1 <- '/data/bm4g15/nalu-stable-exp/csvs/single_layer_task_range_FINAL.csv'
file2 <- '/data/bm4g15/nalu-stable-exp/csvs/SLTR_NALU_sub_units.csv'
file3 <- '/data/bm4g15/nalu-stable-exp/csvs/SLTR_NAU_sub_RegF.csv'
file4 <- '/data/bm4g15/nalu-stable-exp/csvs/SLTR_NAU_add_RegF.csv'
file5 <- '/data/bm4g15/nalu-stable-exp/csvs/SLTR_npu-eps32.csv'
file6 <- '/data/bm4g15/nalu-stable-exp/csvs/SLTR_realnpu-eps32.csv'
save_file <- '/data/bm4g15/nalu-stable-exp/csvs/single_layer_task_range_ALL_FINAL.csv'
ds1 = read_csv(file1)
# Used to filter out the single layer task range results for NAU add and sub and combine the new NAU sub and add results. Do the same for NPU and RealNPU aswell.
ds1 = dplyr::filter(ds1, !grepl("reregualizedlinearnac_op-sub|reregualizedlinearnac_op-add|npu_op|realnpu_op",name))
ds2 = read_csv(file2)
ds3 = read_csv(file3)
ds4 = read_csv(file4)
ds5 = read_csv(file5)
ds6 = read_csv(file6)
combined_ds = rbind(ds1,ds2,ds3,ds4,ds5,ds6)
write_csv(combined_ds, save_file)

