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
    "nips-sltr-in2" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/', 'realnpu_baseline', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/', 'realnpu_modified', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/', 'nru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/', 'sign-nmru', csv_ext, sep = '')
    ),
      list('Real NPU (baseline)', 'Real NPU (modified)', 'NRU', 'NMRU')
    ),
    "nips-sltr-in10" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_baseline', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_modified', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'nru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'sign-nmru', csv_ext, sep = '')
    ),
      list('Real NPU (baseline)', 'Real NPU (modified)', 'NRU', 'NMRU')
    ),
    "nips-sltr-in10-losses-nru" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/', 'nru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'nru_pcc', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'nru_mape', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-losses-realnpu" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_modified', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_modified_pcc', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_modified_mape', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-losses-nmru" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/', 'sign-nmru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'sign-nmru_pcc', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'sign-nmru_mape', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-losses-nmru-correctEps" = npu.csv.merger(list(
      paste('/data/nalms/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2', csv_ext, sep = ''),
      paste('/data/nalms/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_pcc', csv_ext, sep = ''),
      paste('/data/nalms/csvs/NMRU_correctEps/', 'inSize-10_signNMRU-gnc-1_lr-1e-2_mape', csv_ext, sep = '')
    ),
      list('MSE', 'PCC', 'MAPE')
    ),
    "nips-sltr-in10-npu" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_modified', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/npu/', 'realMod', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/npu/', 'realMod_Reg-Wim-l1_clip-wig', csv_ext, sep = '')
    ),
      list('Real NPU (modified)', 'NPU (no constraints)', 'NPU (clip & reg)')
    ),
    "nips-realnpu-L1" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/', 'L1F', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/', 'realnpu_baseline', csv_ext, sep = '')
    ),
      list('L1 off', 'L1 on')
    ),
    "nips-realnpu-L2" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/', 'L1F', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/', 'realnpu_baseline', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/', 'L2', csv_ext, sep = '')
    ),
      list('No reg', 'L1', 'L2')
    ),
    "nips-realnpu-L1_sweep" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/L1_beta_sweep/', '1e-11_1e-9', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/', 'realnpu_baseline', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/L1_beta_sweep/', '1e-8_1e-6', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/L1_beta_sweep/', '1e-7_1e-5', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/L1_beta_sweep/', '1e-5_1e-3', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/L1_beta_sweep/', '1e-3_1e-1', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/L1_beta_sweep/', '1e-1_10', csv_ext, sep = '')
    ),
      list('(1e-11,1e-9)', '(1e-9,1e-7)', '(1e-8,1e-6)', '(1e-7,1e-5)', '(1e-5,1e-3)', '(1e-3,1e-1)', '(1e-1,10)')
    ),
    "nips-realnpu-clipping" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/', 'realnpu_baseline', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/clip/', 'G', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/clip/', 'W', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/clip/', 'GW', csv_ext, sep = '')
    ),
      list('None', 'G', 'W', 'GW')
    ),
    "nips-realnpu-discretisation" = npu.csv.merger(list(
        paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/clip/', 'GW', csv_ext, sep = ''),
        paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/discretisation/', 'G1', csv_ext, sep = ''),
        paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/discretisation/', 'G1-W1', csv_ext, sep = '')
    ),
      list('None','G', 'GW')
    ),
    "nips-realnpu-init" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/realnpu-modifications/discretisation/', 'G1-W1', csv_ext, sep = ''),
      paste('sltr-in2/realnpu-modifications/init/', 'XUC', csv_ext, sep = '')
    ),
      list('Xavier-Uniform', 'Xavier-Uniform Constrained')
    ),
    "nips-in10-nmru-ablation" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/nmru-extra/', 'nmru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/nmru-extra/', 'nmru_gnc-1', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/nmru-extra/', 'sign-nmru_gnc-F', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'sign-nmru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/nmru-extra/', 'sign-nmru_gnc-1_gate-1', csv_ext, sep = '')
    ),
      list('vanilla', 'gnc', 'sign', 'gnc+sign', 'gnc+sign+gating')
    ),
    "nips-in10-nmru-lr" = npu.csv.merger(list(
      paste('/data/nalms/csvs/nmru-extra/', 'lr-1e-3', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'sign-nmru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/nmru-extra/', 'lr-1e-1', csv_ext, sep = '')
    ),
      list('1e-3','1e-2', '1e-1')
    ),
  "nips-in10-nmru-optimiser" = npu.csv.merger(list(
      paste('/data/nalms/csvs/nmru-extra/', 'sgd', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'sign-nmru', csv_ext, sep = '')
    ),
      list('sgd','adam')
    ),
    "nips-in2-nru-lr" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/nru-extra/', 'lr-1e-3', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/nru-extra/', 'lr-1e-2', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/nru-extra/', 'lr-1e-1', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/', 'nru', csv_ext, sep = '')
    ),
      list('1e-3','1e-2','1e-1','1')
    ),
    "nips-in10-nru-separate-mag-sign" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/', 'nru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'nru-sepSign', csv_ext, sep = '')
    ),
      list('together', 'separate')
    ),
    "nips-in10-realnpu-W-reg" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_modified', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/', 'realnpu_modified_W-nau-reg', csv_ext, sep = '')
    ),
      list('{-1,1}', '{-1,0,1}')
    ),
    "nips-in2-distributions" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in2/distributions/', 'realnpu_modified', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/distributions/', 'nru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in2/distributions/', 'sign-nmru', csv_ext, sep = '')
    ),
      list('Real NPU (modified)', 'NRU', 'NMRU')
    ),
    "nips-in10-distributions" = npu.csv.merger(list(
      paste('/data/nalms/csvs/sltr-in10/distributions/', 'realnpu_modified', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/distributions/', 'nru', csv_ext, sep = ''),
      paste('/data/nalms/csvs/sltr-in10/distributions/', 'sign-nmru', csv_ext, sep = '')
    ),
      list('Real NPU (modified)', 'NRU', 'NMRU')
    )
  ))
}








