source('./_expand_name.r')

csv_merger = function(load_files_names, models_name_list) {
  combined_tables <- NULL
  # load tables for each element in the list
  tables <- lapply(load_files_names, read_csv)
  for (idx in 1:length(tables)) {
    t <- ldply(tables[idx], data.frame)  # convert from list to df
    # don't process dfs with no rows - to avoid dists where all configs failed to reach required max step
    if (!empty(t)) {
      # expand the name
      t <- expand.name(t)
      # rename model if names have been given
      if (length(models_name_list)) {
        t$model <- models_name_list[[idx]]      # rename the model name to pre-defined value in list
      }
      # only get common cols if a rbinded table exists (i.e. both dfs to merge do actually have cols)
      if (idx != 1) {
        common_cols <- intersect(colnames(combined_tables), colnames(t))  # get the common columns between the t tables to be merged 
        combined_tables <- rbind(combined_tables[common_cols], t[common_cols])  # add model data to an accumulated table
      } else {
        combined_tables <- rbind(combined_tables, t)  # add model data to an accumulated table
      }      
    }
  }
  return(combined_tables)
}

load_and_merge_csvs = function(lookup.name, single_filepath = NA) {
  csv_ext = '.csv'
  return(switch(
    lookup.name,
    "None" = csv_merger(
      list(single_filepath),
      list('Test')
    ),
    "test" = csv_merger(list(
      paste0(load_folder, 'add_nalmF_fixF', csv_ext),
      paste0(load_folder, 'add_nalmF_fixT', csv_ext),
      paste0(load_folder, 'add_nalmT_fixF', csv_ext)
    ),
      list('covn-fc', 'covn-add', 'convn-nau')
    ),
    "add" = csv_merger(list(
      paste0(load_folder, 'conv-add', csv_ext),
      paste0(load_folder, 'conv-fc', csv_ext),
      paste0(load_folder, 'conv-nau', csv_ext),
      paste0(load_folder, 'conv-add_opt-Adadelta-rho-0.95_lr-1', csv_ext)
    ),
      list('covn-add (Adam)', 'covn-fc', 'conv-nau',  'convn-add (Adadelta)')
    ),
    "mul" = csv_merger(list(
      paste0(load_folder, 'conv-mul', csv_ext),
      paste0(load_folder, 'conv-fc', csv_ext),
      paste0(load_folder, 'conv-nmu', csv_ext),
      paste0(load_folder, 'conv-snmu', csv_ext)
    ),
      list('covn-mul', 'covn-fc', 'convn-nmu', 'conv-snmu')
    ),
####################### MULTIPLICATION  #################################################################
    "static-mnist-mul-isolated" = csv_merger(list(
      paste0(load_folder, '1digit_conv-mul_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-fc_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-nmu_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-snmu', csv_ext),
      paste0(load_folder, '1digit_conv-batch-snmu', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'snmu [1,5]', 'snmu [1,1+1/sd(x)]')
    ),
    "static-mnist-mul-colour-concat" = csv_merger(list(
      paste0(load_folder, 'mul_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'fc_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'nmu_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'snmu_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'batch-snmu_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'snmu [1,5]', 'snmu [1,1+1/sd(x)]')
    ),
    "mul-adadelta" = csv_merger(list(
      paste0(load_folder, 'resnet18-mul-Adadelta', csv_ext),
      paste0(load_folder, 'resnet18-fc-Adadelta', csv_ext),
      paste0(load_folder, 'resnet18-nmu-Adadelta', csv_ext),
      paste0(load_folder, 'resnet18-nmu-AdadeltaR25-35', csv_ext),
      paste0(load_folder, 'resnet18-snmu-Adadelta', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'nmu R25-35', 'snmu')
    ),
    "mul-1digit_conv-Adadelta" = csv_merger(list(
      paste0(load_folder, '1digit_conv-mul', csv_ext),
      paste0(load_folder, '1digit_conv-fc', csv_ext),
      paste0(load_folder, '1digit_conv-nmu', csv_ext),
      paste0(load_folder, '1digit_conv-snmu', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'snmu')
    ),
    "mul-1digit_conv-snmu-dev" = csv_merger(list(
      paste0(load_folder, '1digit_conv-snmu', csv_ext),
      paste0(load_folder, '1digit_conv-snmu_r10-20-10', csv_ext),
      paste0(load_folder, '1digit_conv-snmu_r10-20-100', csv_ext),
      paste0(load_folder, '1digit_conv-snmu-id20', csv_ext),
      paste0(load_folder, '1digit_conv-snmu-id21', csv_ext),
      paste0(load_folder, '1digit_conv-snmu-id22', csv_ext)
    ),
      list('(Adadelta + scheduler)', '(r10-20-10) (Ada.+schd)', '(r10-20-100) (Ada.+schd) (E100)', 
      '(r10-20-100) (Adam+schd)', '(r30-40-100) (Adam+schd(ms=[50]))', '(r30-40-100) (Adam)(E1K)')
    ),
    "mul-1digit_conv-Adam" = csv_merger(list(
      paste0(load_folder, '1digit_conv-mul_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-fc_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-nmu_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-snmu-id22', csv_ext),
      paste0(load_folder, '1digit_conv-batch-snmu-id70', csv_ext)
    ),
      #list('mul', 'fc', 'nmu', 'snmu [1,5]')
      list('mul', 'fc', 'nmu', 'snmu [1,5]', 'snmu [1,1+1/sd(x)]')
    ),
    "mul-spatial-transformer" = csv_merger(list(
      paste0(load_folder, 'st-mul_Adam', csv_ext),
      paste0(load_folder, 'st-snmu_Adam', csv_ext),
      paste0(load_folder, 'st-mul_Adadelta_MSLR-35-75-125', csv_ext),
      paste0(load_folder, 'noConcat-st-mul_Adam', csv_ext),
      paste0(load_folder, 'noConcat-st-mul_Adadelta_MSLR-35-75-125', csv_ext)
    ),
      list('mul (Adam)', 'snmu (Adam)', 'mul (Adadelta & Scd.)', 'mul (noconcat; Adam)', 'mul (noconcat;Adadelta & Scd.)')
    ),
    "mul-colour_no-concat-conv" = csv_merger(list(
      paste0(load_folder, 'ID55-stOrgTask-mul_MSE_Adam-lr0.001_no-concat-conv', csv_ext),
      paste0(load_folder, 'ID58-fc_MSE_Adam-lr0.001_colour_no-concat-conv', csv_ext),
      paste0(load_folder, 'ID59-fc-gnc1_MSE_Adam-lr0.001_colour_no-concat-conv', csv_ext),
      paste0(load_folder, 'ID56-snmu_MSE_Adam-lr0.001_colour_no-concat-conv', csv_ext),
      paste0(load_folder, 'ID57-nmu_MSE_Adam-lr0.001_colour_no-concat-conv', csv_ext)
    ),
      list('mul', 'fc', 'fc (gnc=1)', 'snmu', 'nmu')
    ),
    "mul-TPS-no-concat-conv" = csv_merger(list(
      paste0(load_folder, 'ID60-mul_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'ID63-fc_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'ID62-nmu_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'ID61-snmu_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext),
      paste0(load_folder, 'ID69-batch-snmu_MSE_Adam-lr0.001_TPS-no-concat-conv', csv_ext)
    ),
      list('mul', 'fc', 'nmu', 'snmu [1,5]', 'snmu [1,1+1/sd(x)]')
    ),
####################### ADDITION #################################################################
    "add-1digit_conv-Adam" = csv_merger(list(
      paste0(load_folder, '1digit_conv-add_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-fc_Adam', csv_ext),
      paste0(load_folder, '1digit_conv-nau_Adam', csv_ext)
    ),
      list('mul', 'fc', 'nau (r30-40-100)')
    ),
###################################################################################################
    stop("Key given to csv_merger does not exist!")
  ))
}
