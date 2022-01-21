library(readr)
library(tibble)
library(plyr)
library(dplyr)

#setwd('C:/Users/mistr/Documents/SOTON/PhD/Code/nalu-stable-exp/export/single_layer_task/robustness/')
#source('./_robustness_expand_name.r')

INPUT_SIZE = 10 # input dimension

# load in the python generated indexes and convert them to R indexing (i.e. add 1 to each index col)
process_index_table = function() {
  dat = read_csv('dataset_solution_indexes.csv')
  dat$index.1 = dat$index.1 + 1
  dat$index.2 = dat$index.2 + 1
  return(dat)
}

generate_nau_solutions = function(dataset_info, solutions_tab, operation) {
  seed = dataset_info$seed
  idx1 = dataset_info$index.1
  idx2 = dataset_info$index.2
  gold_weights = double(INPUT_SIZE)
  gold_weights[idx1] = 1.
  if (operation == 'op-add') {
    gold_weights[idx2] = 1.
  } else if (operation == 'op-sub') {
    gold_weights[idx2] = -1.
  }
  op = revalue(operation, operation.full.to.short, warn_missing = FALSE)
  solutions_tab = rbind(solutions_tab, c(model = 'NAU', operation = op, seed = seed,
                                         gold.idxs.1 = idx1, gold.idxs.2 = idx2,
                                         gold.W = list(gold_weights), gold.W_real = NA, gold.gate = NA))
  return(solutions_tab)
}

generate_nmu_solutions = function(dataset_info, solutions_tab, model_name) {
  seed = dataset_info$seed
  idx1 = dataset_info$index.1
  idx2 = dataset_info$index.2
  gold_weights = double(INPUT_SIZE)
  gold_weights[idx1] = 1.
  gold_weights[idx2] = 1.
  op = revalue('op-mul', operation.full.to.short, warn_missing = FALSE)
  solutions_tab = rbind(solutions_tab, c(model = model_name, operation = op, seed = seed,
                                         gold.idxs.1 = idx1, gold.idxs.2 = idx2,
                                         gold.W = list(gold_weights), gold.W_real = NA, gold.gate = NA))
  return(solutions_tab)
}

generate_nmru_solutions = function(dataset_info, solutions_tab, operation, model_name) {
  input_size = INPUT_SIZE * 2

  seed = dataset_info$seed
  idx1 = as.integer(dataset_info$index.1)
  idx2 = as.integer(dataset_info$index.2)

  # add optimal solution
  if(operation == 'op-mul'){
    gold_weights = double(input_size)
    gold_weights[idx1] = 1.
    gold_weights[idx2] = 1.
  } else if(operation == 'op-div'){
    gold_weights = double(input_size)
    gold_weights[idx1] = 1.
    gold_weights[idx2 + input_size / 2] = 1.
  }


  op = revalue(operation, operation.full.to.short, warn_missing=FALSE)
  solutions_tab = rbind(solutions_tab, c(model = model_name, operation = op, seed = seed,
                                         gold.idxs. = c(idx1, idx2),
                                         gold.W = list(gold_weights), gold.W_real = NA, gold.gate = NA))

  ## add solutions which exploit inverse multiplication a * 1/a = 1
  #for (i in 1:(input_size / 2)) {
  #  # only add solutions which exploit the multiplicative identity rule for indexes which are not part of the optimal solution
  #  if (i != idx1 & i != idx2) {
  #    gold_weights = double(input_size)
  #
  #    # set relevant weights to 1
  #    gold_weights[idx1] = 1.
  #    gold_weights[idx2 + input_size / 2] = 1.
  #
  #    # set the multiplicative and division part  for a input element to 1 (so it can canel out)
  #    gold_weights[i] = 1
  #    gold_weights[i + input_size / 2] = 1
  #
  #    solutions_tab = rbind(solutions_tab, c(model = 'NMRU', seed = seed, W = list(gold_weights)))
  #  }
  #}
  #return(list(c(model = 'NMRU', seed = seed, W = gold_weights), c(model = 'sNMRU', seed = seed, W = gold_weights)))
  return(solutions_tab)
}

generate_realnpu_solutions = function(dataset_info, solutions_tab, operation, model_name) {
  input_size = INPUT_SIZE
  seed = dataset_info$seed
  idx1 = as.integer(dataset_info$index.1)
  idx2 = as.integer(dataset_info$index.2)

  create_optimal_solution = function(input_size, idx1, idx2, operation) {
    W_real = double(input_size)
    W_real[idx1] = 1.
    if (operation == 'op-mul') {
      W_real[idx2] = 1.
    } else if (operation == 'op-div') {
      W_real[idx2] = -1.
    }

    gate = double(input_size)
    gate[idx1] = 1.
    gate[idx2] = 1.

    return(list(W_real, gate))
  }

  opt_sol = create_optimal_solution(input_size, idx1, idx2, operation)
  opt_W_real = as.double(unlist(opt_sol[1]))
  opt_gate = as.double(unlist(opt_sol[2]))
  op = revalue(operation, operation.full.to.short, warn_missing=FALSE)
  solutions_tab = rbind(solutions_tab, c(model = model_name, operation = op, seed = seed,
                                          gold.idxs.= c(idx1, idx2),
                                          gold.W = NA, gold.W_real = list(opt_W_real), gold.gate = list(opt_gate)))

  #for (i in 1:input_size) {
  #  if (i != idx1 & i != idx2) {
  #    W_real = opt_W_real
  #    gate = opt_gate
  #
  #    # add row for alternate solution where gate is a 0 so W_real can be anything
  #    W_real[i] = NA
  #    gate[i] = 0
  #    solutions_tab = rbind(solutions_tab, c(model = 'RealNPU', operation = op, seed = seed, W = NA, W_real = list(W_real), gate = list(gate)))
  #
  #    # add row for alternate solution where  W_real is 0 so gate can be anything
  #    W_real[i] = 0
  #    gate[i] = NA
  #    solutions_tab = rbind(solutions_tab, c(model = 'RealNPU', operation = op, seed = seed, W = NA, W_real = list(W_real), gate = list(gate)))
  #
  #  }
  #}

  return(solutions_tab)
}

# creates a csv containing the 'golden' weights given a model, config and seed. The seed determines the indexing of the
# solution
create_golden_solutions = function() {
  index_table = process_index_table()
  solutions_table = rbind()
  for (i in 1:nrow(index_table)) {
    solutions_table = generate_nau_solutions(index_table[i,], solutions_table, 'op-add')
    solutions_table = generate_nau_solutions(index_table[i,], solutions_table, 'op-sub')
    solutions_table = generate_nmu_solutions(index_table[i,], solutions_table, 'NMU')
    solutions_table = generate_nmu_solutions(index_table[i,], solutions_table, 'sNMU')
    solutions_table = generate_nmru_solutions(index_table[i,], solutions_table, 'op-div', 'Sign NMRU')
    solutions_table = generate_nmru_solutions(index_table[i,], solutions_table, 'op-mul', 'Sign NMRU')
    solutions_table = generate_nmru_solutions(index_table[i,], solutions_table, 'op-div', 'Sign sNMRU')
    solutions_table = generate_nmru_solutions(index_table[i,], solutions_table, 'op-mul', 'Sign sNMRU')
    solutions_table = generate_realnpu_solutions(index_table[i,], solutions_table, 'op-mul', 'Real NPU')
    solutions_table = generate_realnpu_solutions(index_table[i,], solutions_table, 'op-div', 'Real NPU')
    solutions_table = generate_realnpu_solutions(index_table[i,], solutions_table, 'op-mul', 'Real NPU (mod)')
    solutions_table = generate_realnpu_solutions(index_table[i,], solutions_table, 'op-div', 'Real NPU (mod)')
  }
  return(data.frame(solutions_table))
}

create_toy_golden_solutions = function() {
  # atm will use npu seed 0 (so 1 set of gold weights)
  #golden_solutions = tibble(model = 'npu', seed = 0, W_real = list(c(1., 1., 0), c(-1., -1., 0)), W_im = list(c(0, 0, 0.), c(0, 0, 0.)), g = list(c(1., 1., 0), c(1., 1., 0)), W = NA)
  #golden_solutions = golden_solutions %>% add_row(model = 'nmu', seed = 0, W = list(c(1., 1., 0), c(-1., -1., 0)))
  golden_solutions = golden_solutions %>% add_row(model = 'npu', seed = 1, W_real = list(c(1., 1., 0)), W_im = list(c(0, 0, 0.)), g = list(c(1., 1., 0)), W = NA)
  return(golden_solutions)
}

#sols = create_golden_solutions()
