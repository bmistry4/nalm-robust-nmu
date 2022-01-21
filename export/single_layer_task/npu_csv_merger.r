# Use in range.r (just paste in the code and replace the file that becomes dat like how it's done in the function_task_static_ranges).
# Reads, and renames and combines the different NPU csv results 

#######################################################################################
## Read in each models csv results and update model names while merging them into one dat
## list of filepaths to the final_metrics results for each different model. (These results will be combined onto the same plot)
#files.list = list(
#  paste(load_folder, 'SLTR_npu-eps32', csv_ext, sep = ''),
#  paste(load_folder, 'SLTR_npu-mul-eps32-regS1e-5E1e-4', csv_ext, sep = ''),
#  paste(load_folder, 'SLTR_npu-eps32-regF-lr5e-3', csv_ext, sep = ''),
#  paste(load_folder, 'SLTR_npu-eps32-regF-lr1e-3', csv_ext, sep = ''),
#  paste(load_folder, 'SLTR_npu-eps32-lr1e-3', csv_ext, sep = '')
#)
#
## Name of the models to associate with each file in the above list. Currently they will all be named NMU.
#models.list = list('eps32', 'R:1e-5_1e-4', 'R:F lr:5e-3', 'R:F lr:1e-3', 'lr:1e-3')
#
## read in each file, rename the model to correct name, and concat all the tables row-wise
#merge.csvs = function(load.files.names, model.names) {
#  combined.tables = NULL
#  # load tables for each element in the list AND EXPAND THEM
#  tables <- lapply(lapply(load.files.names, read_csv), expand.name)
#  for (idx in 1:length(tables)) {
#    t <- ldply(tables[idx], data.frame)  # convert from list to df
#    t$model <- model.names[[idx]]      # rename the model name to pre-defined value in list
#    combined.tables = rbind(combined.tables, t)  # add model data to an accumulated table
#  }
#  return(combined.tables)
#}
#
#csvs.combined = merge.csvs(files.list, models.list)
## dat needs model col to be a factor (because different models = different levels).
## Without this line, you can't drop levels when plotting
#csvs.combined$model <- as.factor(as.vector(csvs.combined$model))
#######################################################################################

npu.csv.merger = function (files.list, models.list) {
  # read in each file, rename the model to correct name, and concat all the tables row-wise
  merge.csvs = function(load.files.names, model.names) {
    combined.tables <- NULL
    # load tables for each element in the list AND EXPAND THEM
    tables <- lapply(lapply(load.files.names, read_csv), expand.name)
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

load.and.merge.csvs = function(lookup.name) {
  return(switch(
    lookup.name,
    "None" = expand.name(read_csv(name.file)),
    "RealNPU_initial_variations" = npu.csv.merger(list(
      paste(load_folder, 'SLTR_realnpu-eps32', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-r-L1F', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-W_clip', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-G_clip_RENAMED', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-r-heim-G1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-r-madsen-G10', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-r-heim-W1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-r-madsen-W10', csv_ext, sep = '')
    ),
      list('baseline l1', 'no l1', 'W clip', 'G clip', 'WG clip', 'heim-G1', 'madsen-G10', 'heim-W1', 'madsen-W10')
    ),
    "RealNPU_L1_sweep" = npu.csv.merger(list(
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-11_1e-9', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-8_1e-6', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_realnpu-eps32', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-7_1e-5', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-5_1e-3', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-3_1e-1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-1_10', csv_ext, sep = '')
    ),
      list('(1e-11,1e-9)', '(1e-8,1e-6)', '(1e-9,1e-7)', '(1e-7,1e-5)', '(1e-5,1e-3)', '(1e-3,1e-1)', '(1e-1,10)')
    ),
    "RealNPU_M_reg_sweep" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_M-S10000-E20000-G10', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_M-S20000-E30000-G10', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_M-S30000-E40000-G10', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_M-S40000-E50000-G10', csv_ext, sep = '')
    ),
      list('baseline', '10K-20K', '20K-30K', '30K-40K', '40K-50K')
    ),
    "RealNPU_M_reg_sweep_larger" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_M-S30000-E40000-G1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_M-S40000-E50000-G1', csv_ext, sep = '')
    ),
      list('baseline', '30K-40K', '40K-50K')
    ),
    "RealNPU_G1vs10" = npu.csv.merger(list(
        paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = ''),
        paste(load_folder, 'SLTR_RealNPU_M-S40000-E50000-G1', csv_ext, sep = ''),
        paste(load_folder, 'SLTR_RealNPU_M-S40000-E50000-G10', csv_ext, sep = '')
      ),
      list('baseline', 'G1:40K-50K', 'G10:40K-50K')
    ),
    "RealNPU_M-G&W" = npu.csv.merger(list(
        paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = ''),
        paste(load_folder, 'SLTR_RealNPU_M-S40000-E50000-G1', csv_ext, sep = ''),
        paste(load_folder, 'SLTR_RealNPU_M-S40000-E50000-G1-W1', csv_ext, sep = '')
    ),
      list('baseline','G1:40K-50K', 'G1,W1:40K-50K')
    ),
    "RealNPU_init" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_M-S40000-E50000-G1-W1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G5-W2-WrI_xuc', csv_ext, sep = '')
    ),
      list('baseline', 'init:xavier-uniform', 'init:NAU,G1,W1', 'init:NAU,G5,W2')
    ),
    "RealNPU_modified_lr_sweep" = npu.csv.merger(list(
      paste(load_folder, 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc-lr_5e-3', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc-lr_1e-2', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc-lr_0.1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc-lr_5e-1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc-lr_1', csv_ext, sep = '')
    ),
      list('lr:5e-3', 'lr:1e-2', 'lr:1e-1', 'lr:5e-1', 'lr:1')
    ),
    "log-interval_nru" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'nru_log-10_tanh1000_lr-1', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-identity-conversion-approx-1000-tanh1000_lr-1', csv_ext, sep = '')
    ),
      list('10', '1000')
    ),
    "log-interval_nru_pcc" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'nru_log-10_tanh1000_lr-1_pcc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'nru_log-1000_tanh1000_lr-1_pcc', csv_ext, sep = '')
    ),
      list('10', '1000')
    ),
    "log-interval_realnpu_mod" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'RealNPU_log-10_clip-WG_M-S40K-E50K-G1-W1_WrI-xuc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc', csv_ext, sep = '')
    ),
      list('10', '1000')
    ),
    "log-interval_realnpu_mod_pcc" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'RealNPU_log-10_clip-WG_M-S40K-E50K-G1-W1_WrI-xuc_pcc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_L1_WG-clip_M-S40000-E50000_G1_W1_WrI-xuc_pcc', csv_ext, sep = '')
    ),
      list('10', '1000')
    ),
    "log-interval_realnpu_baseline" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'RealNPU_log-10_baseline', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_npu-eps32', csv_ext, sep = '')
    ),
      list('10', '1000')
    ),
    "log-interval_realnpu_baseline_pcc" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'RealNPU_log-10_baseline_pcc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'RealNPU_log-1000_baseline_pcc', csv_ext, sep = '')
    ),
      list('10', '1000')
    ),
    "log-interval_mixedSigns_realnpu_mod" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_log-intervals/', 'RealNPU_log-10_mixedSigns_clip-WG_M-S40K-E50K-G1-W1_WrI-xuc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc-lr_5e-3', csv_ext, sep = '')
    ),
      list('10', '1000')
    ),    
    "nips-sltr-in2" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_realnpu-eps32', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc', csv_ext, sep = ''), 
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-identity-conversion-approx-1000-tanh1000_lr-1', csv_ext, sep = ''), 
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-2_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = '')
    ),
      list('Real NPU (baseline)', 'Real NPU (modified)', 'NRU', 'NMRU')
    ),
    "nips-sltr-in10" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_baseline', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_WG_clip-M-S50K-E75K-G1-W1-WrI_xuc', csv_ext, sep = ''), 
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-absApprox-tanh1000_lr-1e-3_inSize-10', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = '')
    ),
      list('Real NPU (baseline)', 'Real NPU (modified)', 'NRU', 'NMRU')
    ),
    "nips-sltr-in10-losses-nru" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-tanh1000_lr-1e-3_inSize-10_E100k_Reg-50k-75k', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-tanh1000_lr-1e-3_inSize-10_E100k_Reg-50k-75k_eps-clamp-pcc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-tanh1000_inSize-10_E100k_Reg-50k-75k_mape', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-losses-realnpu" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_WG_clip-M-S50K-E75K-G1-W1-WrI_xuc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_mod_pcc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_mod_mape_E100K', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-losses-nmru" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_pcc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_mape', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-losses-nmru-correctEps" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_pcc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_mape', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-npu" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_WG_clip-M-S50K-E75K-G1-W1-WrI_xuc', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NPU_inSize-10/', 'NPU_in10_realMod', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NPU_inSize-10/', 'NPU_in10_realMod_Reg-Wim-l1_clip-wig', csv_ext, sep = '')
    ),
      list('Real NPU (modified)', 'NPU (no constraints)', 'NPU (clip & reg)')
    ),
    "nips-realnpu-L1" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-r-L1F', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_realnpu-eps32', csv_ext, sep = '')
    ),
      list('L1 off', 'L1 on')
    ),
    "nips-realnpu-L2" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-r-L1F', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_realnpu-eps32', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-r-L2', csv_ext, sep = '')
    ),
      list('No reg', 'L1', 'L2')
    ),
    "nips-realnpu-L1_sweep" = npu.csv.merger(list(
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-11_1e-9', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_realnpu-eps32', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-8_1e-6', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-7_1e-5', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-5_1e-3', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-3_1e-1', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_L1_sweep_1e-1_10', csv_ext, sep = '')
    ),
      list('(1e-11,1e-9)', '(1e-9,1e-7)', '(1e-8,1e-6)', '(1e-7,1e-5)', '(1e-5,1e-3)', '(1e-3,1e-1)', '(1e-1,10)')
    ),
    "nips-realnpu-clipping" = npu.csv.merger(list(
      paste(load_folder, 'SLTR_realnpu-eps32', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-G_clip_RENAMED', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-W_clip', csv_ext, sep = ''),
      paste(load_folder, 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = '')
    ),
      list('None', 'G', 'W', 'GW')
    ),
    "nips-realnpu-discretisation" = npu.csv.merger(list(
        paste('/data/bm4g15/nalu-stable-exp/csvs/', 'SLTR_RealNPU_variations-WG_clip', csv_ext, sep = ''),
        paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_M-S40000-E50000-G1', csv_ext, sep = ''),
        paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_M-S40000-E50000-G1-W1', csv_ext, sep = '')
    ),
      list('None', 'G', 'GW')
    ),
    "nips-realnpu-init" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_M-S40000-E50000-G1-W1', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU/', 'SLTR_RealNPU_WG_clip-M-S40000-E50000-G1-W1-WrI_xuc', csv_ext, sep = '')
    ),
      list('Xavier-Uniform', 'Xavier-Uniform Constrained')
    ),
    "nips-in10-nmru-ablation" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_epsReciprocal_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_epsReciprocal_gradNormClip-1_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-F_lr-1e-2_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_gate-10-input_E100K', csv_ext, sep = '')
    ),
      list('vanilla', 'gnc', 'sign', 'gnc+sign', 'gnc+sign+gating')
    ),
    "nips-in10-nmru-lr" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-1_E100K', csv_ext, sep = '')
    ),
      list('1e-3','1e-2', '1e-1')
    ),
    "nips-in10-nmru-correctEps-lr" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-3', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-1', csv_ext, sep = '')
    ),
      list('1e-3','1e-2', '1e-1')
    ),
  "nips-in10-nmru-optimiser" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_sgd_E100K', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_NMRU/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_E100K', csv_ext, sep = '')
    ),
      list('sgd','adam')
    ),
    "nips-in10-nmru-correctEps-ablation" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_NMRU_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_NMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-F_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_gate-1', csv_ext, sep = '')
    ),
      list('vanilla', 'gnc', 'sign', 'gnc+sign', 'gnc+sign+gating')
    ),
    "nips-in10-nmru-correctEps-lr" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-3', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-1', csv_ext, sep = '')
    ),
      list('1e-3','1e-2', '1e-1')
    ),
  "nips-in10-nmru-correctEps-optimiser" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_sgd', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = '')
    ),
      list('sgd','adam')
    ),
    "nips-in2-nru-lr" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-identity-conversion-approx-1000-tanh1000_absApprox', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-identity-conversion-approx-1000-tanh1000_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-identity-conversion-approx-1000-tanh1000_lr-1e-1', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-identity-conversion-approx-1000-tanh1000_lr-1', csv_ext, sep = '')
    ),
      list('1e-3','1e-2','1e-1','1')
    ),
    "nips-in10-nru-separate-mag-sign" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_NRU-tanh1000_lr-1e-3_inSize-10_E100k_Reg-50k-75k', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/NRU/', 'SLTR_tanh1000-SepSigns_lr-1e-3_inSize-10', csv_ext, sep = '')
    ),
      list('together', 'separate')
    ),
    "nips-in10-realnpu-W-reg" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_WG_clip-M-S50K-E75K-G1-W1-WrI_xuc', csv_ext, sep = ''), 
      paste('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_inSize-10/', 'SLTR_RealNPU_in10_mod_W-nau-reg', csv_ext, sep = '')
    ),
      list('{-1,1}', '{-1,0,1}')
    ),
    "nips-sltr-in2-benford" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'RealNPU-mod_inSize2_benford', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-2_NRU_lr-1', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-2_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = '') 
      
    ),
      list('Real NPU (modified)', 'NRU', 'NMRU')
    ),
    "nips-sltr-in10-benford" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'RealNPU-mod_inSize10_benford', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-10_NRU_lr-1e-3', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-10_signNMRU-gnc-1_lr-1e-2_stochastic-0.1-0.5', csv_ext, sep = '')

    ),
      list('Real NPU (modified)', 'NRU', 'NMRU', 'Stochastic NMRU')
    ),
    "nips-sltr-in2-rebuttal" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'RealNPU-mod_inSize2', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'inSize-2_NRU_lr-1', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'inSize-2_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'RealNPU-mod_inSize2_benford', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-2_NRU_lr-1', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-2_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = '')  
      
    ),
      list('Real NPU (modified)', 'NRU', 'NMRU',
           'Real NPU (modified)', 'NRU', 'NMRU'
      )
    ),
    "nips-sltr-in10-rebuttal" = npu.csv.merger(list(
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'RealNPU-mod_inSize10', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'inSize-10_NRU_lr-1e-3', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'RealNPU-mod_inSize10_benford', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-10_NRU_lr-1e-3', csv_ext, sep = ''),
      paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'benford_inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = '')
      #paste('/data/bm4g15/nalu-stable-exp/csvs/neurips2021_rebuttal/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_stochastic-0.1-0.5', csv_ext, sep = '')

    ),
      list('Real NPU (modified)', 'NRU', 'NMRU', 
           'Real NPU (modified)', 'NRU', 'NMRU'
      )
    )
  ))
}
# example usage








