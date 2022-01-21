csv.merger = function(files.list, models.list) {
  # read in each file, rename the model to correct name, and concat all the tables row-wise
  merge.csvs = function(load.files.names, model.names) {
    combined.tables <- NULL
    # load tables for each element in the list AND EXPAND THEM
    tables <- lapply(lapply(load.files.names, read_csv), expand.name)
    for (idx in 1:length(tables)) {
      t <- ldply(tables[idx], data.frame)  # convert from list to df
      # create model col only if a list of model names has been given. Otherwise will use predefined names from _single_layer_task_expand_name.r
      if (length(model.names) != 0) {
        t$model <- model.names[[idx]]      # rename the model name to pre-defined value in list
      }
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
    "fts-icml2020" = csv.merger(list(
      paste0(load_folder, 'FTS_NAU_NMU_ranges_baseline', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges_noise-1-5', csv_ext),
      paste0(load_folder, 'FTS_nau-Nnmu-epsPcc-750K-mse', csv_ext),
      paste0(load_folder, 'FTS_beta-nau-Nnmu_ranges-pcc-750K-mse', csv_ext)
    ),
      list('NMU', 'sNMU', 'PCC-MSE', 'bNAU')
    ),
    "fts-icml2020-rebuttal" = csv.merger(list(
      paste0(load_folder, 'FTS_nau-Nnmu-epsPcc-750K-mse', csv_ext),
      paste0(load_folder, 'FTS_nau-nmu-pcc750Kmse', csv_ext),
      paste0(load_folder, 'FTS_nau-nmu-pcc', csv_ext)
    ),
      list('NAU-sNMU PCC-MSE', 'NAU-NMU PCC-MSE', 'NAU-NMU PCC')
    ),
    "fts-2021" = csv.merger(list(
      paste0(load_folder, 'FTS_NAU_NMU_ranges_baseline', csv_ext),
      paste0(load_folder, 'FTS_nau-nmu-pcc', csv_ext),
      paste0(load_folder, 'FTS_nau-nmu-pcc750Kmse', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges_noise-1-5', csv_ext),
      paste0(load_folder, 'FTS_nau-Nnmu-epsPcc-750K-mse', csv_ext)
    ),
      list('NMU', 'NMU (PCC)', 'NMU (PCC-MSE)', 'sNMU (MSE)', 'sNMU (PCC-MSE)')
    ),
    "fts-2021-final" = csv.merger(list(
      paste0(load_folder, 'FTS_NAU_NMU_ranges_baseline', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges_noise-1-5', csv_ext)
    ),
      list('NMU', 'sNMU')
    ),
    "fts-noise" = csv.merger(list(
      paste0(load_folder, 'FTS_NAU_NMU_ranges_noise-1-5', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/batch_sNMU', csv_ext),
      paste0(load_folder, 'FTS_nau-stg_nmu', csv_ext)
    ),
      list('sNMU', 'Batch sNMU', 'stg-NMU')
    ),
    "fts-mape" = csv.merger(list(
      paste0(load_folder, 'FTS_NAU_NMU_ranges_baseline', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/FTS_nau-nmu-mape', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/FTS_nau-snmu-mape', csv_ext)
    ),
      list('nmu (mse)', 'nmu (mape)', 'snmu (mape)')
    ),
    "fts-snmu-noise-ranges" = csv.merger(list(
      paste0(load_folder, 'FTS_NAU_NMU_ranges/sNMU-noise-ranges/FTS_nau-snmu_0.01-0.05', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/sNMU-noise-ranges/FTS_nau-snmu_0.1-0.5', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/sNMU-noise-ranges/FTS_nau-snmu_1-2', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges_noise-1-5', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/sNMU-noise-ranges/FTS_nau-snmu_5-10', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/sNMU-noise-ranges/FTS_nau-snmu_10-20', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/batch_sNMU', csv_ext)
    ),
      list('[0.01,0.05]', '[0.1,0.5]', '[1,2]', '[1,5]', '[5,10]', '[10,20]', '[1,1+1/sd(x)]')
    ),
    "fts-snmu-test" = csv.merger(list(
      paste0(load_folder, 'FTS_NAU_NMU_ranges_noise-1-5', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/sNMU-noise-ranges/FTS_nau-snmu_5-10', csv_ext),
      paste0(load_folder, 'FTS_NAU_NMU_ranges/sNMU-noise-ranges/FTS_nau-snmu_10-20', csv_ext)
    ),
      list('[1,5]', '[5,10]', '[10,20]')
    ),
    stop("Key given to csv_merger does not exist!")
  ))
}









