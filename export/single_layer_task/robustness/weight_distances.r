#rm(list = ls())
#setwd('C:/Users/mistr/Documents/SOTON/PhD/Code/nalu-stable-exp/export/single_layer_task/robustness/')

#library(ggplot2)
#library(plyr)
#library(dplyr)
#library(readr)
#library(tibble)

source('./_robustness_expand_name.r')
source('./generate_solutions.r')


#model_name = 'None' # 'Real NPU (mod)'
##load_file = 'toy_data_expanded/nau.csv'
#load_file =  'nmru-div_truncated-normal-0-1--5-5_expanded_last_step.csv' # 'hparam_search_nau-add_truncated-normal-0-1--5-5.csv' #  'nmru-div_truncated-normal-0-1--5-5_expanded_last_step.csv'#
#coarseness = 'config'
#violin_output = 'violin_plot.pdf'
#ci_output = 'ci_plot.pdf'

csv_merger = function(load_files_names, distributions_list) {
  combined_tables <- NULL
  # load tables for each element in the list
  tables <- lapply(load_files_names, read_csv)
  for (idx in 1:length(tables)) {
    t <- ldply(tables[idx], data.frame)  # convert from list to df
    t$distribution.id <- distributions_list[[idx]]      # rename the model name to pre-defined value in list
    combined_tables <- rbind(combined_tables, t)  # add model data to an accumulated table
  }
  return(combined_tables)
}

# TODO change to specific epoch
# get the last epoch data for each config
largest.step.row = function(dat) {
  return(dat %>%
           group_by(name) %>%
           slice_tail() %>%
           ungroup())
}

# Get filter out each experiement to show the results for a specific step (i.e. epoch)
filter.by.step = function(dat, step_value = 100000) {
  return(dat %>%
           group_by(name) %>%
           filter(step == step_value) %>%
           ungroup()
  )
}

#dat = read_csv(load_file)
#dat = largest.step.row(expand.name(read_csv(load_file)))  # TODO: use if NAU-add
#write.csv(largest.step.row(dat), paste0(load_folder, base_filename, '_expanded_last_step.csv'))

# get all the columns containing .param.
filter_param_columns = function(table, prefix = "param.", parameter = NA) {
  if (is.na(parameter)) {
    return(select(table, contains(prefix)))
  } else {
    return(select(table, contains(paste0(prefix, parameter, '.'))))
  }
}

avg_l1_dist = function(row) {
  model_W = as.vector(filter_param_columns(row, parameter = 'W'))
  gold_W = row$gold.W[[1]]
  return(as.numeric(rowMeans(abs(model_W - gold_W))))

}

avg_l1_dist_realnpu = function(row) {
  eps = 1e-5
  model_W_real = as.vector(filter_param_columns(row, parameter = 'W_real'))
  model_gate = as.vector(filter_param_columns(row, parameter = 'gate'))
  gold_W_real = row$gold.W_real[[1]]
  gold_gate = row$gold.gate[[1]]

  dist = 0.
  for (i in 1:length(model_W_real)) {
    if (!(i == row$gold.idxs.1 | i == row$gold.idxs.2)) {
      # if W_real ~= 0 then the corresponding gate value can be anything
      if (abs(model_W_real[i]) <= eps) {
        dist = dist + abs(model_W_real[i] - gold_W_real[i])
      }
        # if gate ~= 0 then the corresponding W_real value can be anything
      else if (model_gate[i] <= eps) {
        dist = dist + abs(model_gate[i] - gold_gate[i])
        # if neither gate or weight is close the 0 then the L1 distance is calculated
      } else {
        dist = dist +
          abs(model_W_real[i] - gold_W_real[i]) +
          abs(model_gate[i] - gold_gate[i])
      }
      # deal with indexes corresponding to a relevant input element
    } else {
      dist = dist +
        abs(model_W_real[i] - gold_W_real[i]) +
        abs(model_gate[i] - gold_gate[i])
    }
  }
  return(as.numeric(dist / (length(model_gate) + length(model_W_real))))
}

avg_l1_dist_nmru = function(row) {
  model_W = as.vector(filter_param_columns(row, parameter = 'W'))
  gold_W = row$gold.W[[1]]
  eps = 1e-5

  dist = 0.
  # go through the weights relevant for the mul part (and deal with the reciprocal part in parallel)
  for (i in 1:(length(model_W)/2)) {
    # check if the current index corresponds to a weights for a irrelevant input
    if (!(i == row$gold.idxs.1 | i == row$gold.idxs.2)) {
          # if the inverse rule (x_i * 1/x_i) is not met, penalise the weights which should be 0 (i.e. those corresponding to irrelevant inputs)
      if (!(model_W[i] >= (1-eps) & model_W[length(model_W)/2 + i] >= (1-eps))) {
        dist = dist +
          abs(model_W[i] - gold_W[i]) +
          abs(model_W[length(model_W)/2 + i] - gold_W[length(model_W)/2 + i])
      }
      # calc distrance for a relevant weight (which should be one) and it mul/recip counterpart (which should be 0)
    }else{
      dist = dist +
          abs(model_W[i] - gold_W[i]) +
          abs(model_W[length(model_W)/2 + i] - gold_W[length(model_W)/2 + i])
    }
  }
  return(as.numeric(dist / length(model_W)))
}

lookup_model_distance = function(row) {
  return(switch(
    as.character(row$model),
    'NAU' = avg_l1_dist(row),
    'Real NPU' = avg_l1_dist_realnpu(row),
    'Real NPU (mod)' = avg_l1_dist_realnpu(row),
    'Sign NMRU' = avg_l1_dist_nmru(row),
    'Sign sNMRU' = avg_l1_dist_nmru(row),
    'NMU' = avg_l1_dist(row),
    'sNMU' = avg_l1_dist(row),
    stop("Invalid model name")
  ))
}

generate_config_id = function (row) {
  # prefix which all models will have
  #base.id = paste0("Dist:", row$train.distribution, "-", row$interpolation.range, " LR:", row$learning.rate)
  base.id = paste0("LR:", row$learning.rate)

  return(switch(
    as.character(row$model),
    'NAU' = paste0(base.id,
              " Reg:", row$regualizer, "-", row$regualizer.scaling.start, "-", row$regualizer.scaling.end),
    'NMU' = paste0(base.id,
              " Reg:", row$regualizer, "-", row$regualizer.scaling.start, "-", row$regualizer.scaling.end),
    'sNMU' = paste0(base.id,
              " Reg:", row$regualizer, "-", row$regualizer.scaling.start, "-", row$regualizer.scaling.end,
              " Noise:", row$noise.range),
    'Sign NMRU' = paste0(base.id,
              " Reg:", row$regualizer, "-", row$regualizer.scaling.start, "-", row$regualizer.scaling.end),
    'Sign sNMRU' = paste0(base.id,
              " Reg:", row$regualizer, "-", row$regualizer.scaling.start, "-", row$regualizer.scaling.end,
              " Noise:", row$noise.range),
    'Real NPU' = paste0(base.id,
              " Beta-reg:", row$regualizer.beta.start, "-", row$regualizer.beta.end),
    'Real NPU (mod)' = paste0(base.id,
             " Reg:", row$regualizer.npu.W, "-", row$regualizer.scaling.start, "-", row$regualizer.scaling.end,
             " Beta reg:", row$regualizer.beta.start, "-", row$regualizer.beta.end),
    stop("Invalid model name")
  ))
}

# works with coarseness='distribution'
plot_violin = function(plot_data) {
  return(
  # TODO - x = config.id if doing individual plots per dist and distribution.id for all-distributions
  ggplot(plot_data,
         aes(x = distribution.id,
             y = distance)) +
    scale_y_log10(oob = scales::squish_infinite) +
    geom_violin(fill = "cornflowerblue") +
    geom_boxplot(width = .05,
                 #fill = "orange",
                 outlier.color = "orange",
                 outlier.size = 0.1, 
                 alpha = 0.4) +
    labs(y = 'Average distance', x = 'Distribution') +
    theme(plot.margin = unit(c(5.5, 10.5, 5.5, 5.5), "points")) +
    geom_hline(yintercept = 1e-5, color = 'red', size = 0.01) #+
  #coord_flip()
  #labs(title = "Average Weight Distances from Golden Solutions")
  )
}

# works with coarseness='config'
# FIXME -> order by lr and other cols
plot_scatter_with_ci = function(plot_data) {
  return(
    ggplot(plot_data,
           aes(x = config.id,
               y = mean
           )) +
      scale_y_log10() +
      geom_point(size = 0.5) +
      # don't plot the lower CI if the (mean - CI) is > 0, otherwise the log10 scale  will not work 
      geom_errorbar(aes(ymin = mean - (mean - ci > 0 )*ci,
                        ymax = mean + ci),
                    width = .1, ) +
      labs(y = 'Average distance', x = 'Configuration') +
      theme(plot.margin = unit(c(5.5, 10.5, 5.5, 5.5), "points"), text = element_text(size = 5)) +
      geom_hline(yintercept = 1e-5, color = 'red', size = 0.01) +
      coord_flip()
  )
}

distances_summary = function(distances_df)
  # 95% confidence intervals using a t-test
  return(distances_df %>%
           summarize(n = n(),
                     mean = mean(distance),
                     sd = sd(distance),
                     se = sd / sqrt(n),
                     ci = qt(0.975, df = n - 1) * sd / sqrt(n))
  )


group_data_by_coarseness = function(dat, coarsenesss_option) {
  if (coarsenesss_option == 'distribution') {
    return(dat %>% group_by(model, train.distribution, interpolation.range))
  }else if (coarsenesss_option == 'config') {
    return(
      dat %>% group_by(config.id)
    )
  }else if (coarsenesss_option == 'all-distributions') {
    return(dat %>% group_by(distribution.id))
  }
}

#print("R Script completed.")
