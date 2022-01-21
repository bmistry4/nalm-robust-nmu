rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)

# ( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )
simulate.mse = function (epsilon, samples, operation, simple, input.size, subset.ratio, overlap.ratio, range.a, range.b, range.mirror) {
  # matrix = (data, row, cols)
  # runif = generate random numbers of len of the 1st arg with ranges a to b
  X = matrix(runif(samples*input.size, range.a, range.b), samples, input.size)

  if (range.mirror) {
    # rbinom(n (|obvs|), size, prob) = rbinom(number of runs, ntrials, success prob)
    # rbinom(400, size = 10, prob = 0.2) = gives the results of 400 runs of 10 coin flips each, returning the number of successes in each run.
    # creates a sign matrix (of -1 and 1s) using a binomial dist
    X.sign = matrix(rbinom(samples*input.size, 1, 0.5), samples, input.size) * 2 - 1
    X = X * X.sign
  }

  evaluate = function (W) {
    # matrix mul
    Z = X %*% W
    if (operation == 'add') {
      Y = Z[,1] + Z[,2]
    } else if (operation == 'sub') {
      Y = Z[,1] - Z[,2]
    } else if (operation == 'mul') {
      Y = Z[,1] * Z[,2]
    } else if (operation == 'div') {
      Y = Z[,1] / Z[,2]
    } else if (operation == 'squared') {
      Y = Z[,1] * Z[,1]
    } else if (operation == 'root') {
      Y = sqrt(Z[,1])
    }
    return(Y)
  }
  
  target.matrix = function (epsilon) {
    if (simple) {
      a.start = 1
      a.end = 2
      b.start = 1
      b.end = 4
    } else {
      subset.size = floor(subset.ratio * input.size)
      overlap.size = floor(overlap.ratio * subset.size)
      
      a.start = 1
      a.end = a.start + subset.size
      b.start = a.end - overlap.size
      b.end = b.start + subset.size
    }
    
    W = matrix(0 + epsilon, input.size, 2)
    W[a.start:a.end, 1] = 1 - epsilon
    W[b.start:b.end, 2] = 1 - epsilon
    return(W)
  }

  W.y = target.matrix(epsilon)
  W.t = target.matrix(0)
  errors = (evaluate(W.y) - evaluate(W.t))**2
  
  return(mean(errors))
}

cases = rbind()
cases = rbind(cases,
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-40, range.b=-20, range.mirror=F),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-6, range.b=-2, range.mirror=F),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-6.1, range.b=-1.2, range.mirror=F),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-2, range.b=-0.2, range.mirror=F),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=2, range.b=6, range.mirror=T),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=0.2, range.b=2, range.mirror=F),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=2, range.b=6, range.mirror=F),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=1.2, range.b=6, range.mirror=F),
  c(parameter='extrapolation.range', operation='mul', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=20, range.b=40, range.mirror=F)
)

cases = rbind(cases,
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-40, range.b=-20, range.mirror=F),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-6, range.b=-2, range.mirror=F),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-6.1, range.b=-1.2, range.mirror=F),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=-2, range.b=-0.2, range.mirror=F),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=2, range.b=6, range.mirror=T),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=0.2, range.b=2, range.mirror=F),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=2, range.b=6, range.mirror=F),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=1.2, range.b=6, range.mirror=F),
  c(parameter='extrapolation.range', operation='div', simple=F, input.size=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=20, range.b=40, range.mirror=F)
)

eps = data.frame(rbind(
  c(operation='mul', epsilon=0.00001),
  c(operation='add', epsilon=0.00001),
  c(operation='sub', epsilon=0.00001),
  c(operation='div', epsilon=0.00001),
  c(operation='squared', epsilon=0.00001),
  c(operation='root', epsilon=0.00001)
))

mse = data.frame(cases) %>%
  merge(eps) %>%
  mutate(
    simple=as.logical(as.character(simple)),
    input.size=as.integer(as.character(input.size)),
    subset.ratio=as.numeric(as.character(subset.ratio)),
    overlap.ratio=as.numeric(as.character(overlap.ratio)),
    range.a=as.numeric(as.character(range.a)),
    range.b=as.numeric(as.character(range.b)),
    range.mirror=as.logical(as.character(range.mirror)),
    epsilon=as.numeric(as.character(epsilon))
  ) %>%
  rowwise() %>%
  mutate(
    threshold=simulate.mse(epsilon, 1000000, operation, simple, input.size, subset.ratio, overlap.ratio, range.a, range.b, range.mirror),
    extrapolation.range=ifelse(range.mirror, paste0('U[-',range.b,',-',range.a,'] ∪ U[',range.a,',',range.b,']'), paste0('U[',range.a,',',range.b,']')),
    operation=paste0('op-', operation)
  )


write.csv(mse, file="exp_setups.csv", row.names=F)
#write.csv(mse, file="../results/function_task_static_mse_expectation.csv", row.names=F)

