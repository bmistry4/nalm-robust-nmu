rm(list = ls())
# setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)

# ( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )
cases = rbind()
for (operation in c('add', 'sub', 'mul', 'div')){
  cases = rbind(cases,
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-40, range.b=-20, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-6, range.b=-2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-6.1, range.b=-1.2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-2, range.b=-0.2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=2, range.b=6, range.mirror=T),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0.2, range.b=2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=2, range.b=6, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=1.2, range.b=6, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=20, range.b=40, range.mirror=F)
  )
}

for (operation in c('div')){
  cases = rbind(cases,
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=100, range.b=1000, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-6, range.b=-2, range.mirror=T)  
  )
}

for (operation in c('div')){
  cases = rbind(cases,
    c(parameter='zero.range.hard', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.hard', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.hard', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.hard', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.hard', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=1.0, range.mirror=F)
  )
}

for (operation in c('reciprocal')){
  cases = rbind(cases,
    c(parameter='zero.range.easy', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.easy', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.easy', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.easy', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.easy', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=1, range.mirror=F)
  )
}

for (operation in c('reciprocal')){
  cases = rbind(cases,
    c(parameter='zero.range.medium', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.medium', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.medium', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.medium', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.medium', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=1, range.mirror=F)
  )
}

for (operation in c('div')){
  cases = rbind(cases,
    c(parameter='zero.range.hard.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.hard.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.hard.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.hard.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.hard.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=1.0, range.mirror=F)
  )
}

for (operation in c('reciprocal')){
  cases = rbind(cases,
    c(parameter='zero.range.easy.realnpu', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.easy.realnpu', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.easy.realnpu', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.easy.realnpu', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.easy.realnpu', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=1, range.mirror=F)
  )
}

for (operation in c('reciprocal')){
  cases = rbind(cases,
    c(parameter='zero.range.medium.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.medium.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.medium.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.medium.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.medium.realnpu', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=1, range.mirror=F)
  )
}

for (operation in c('div')){
  cases = rbind(cases,
    c(parameter='zero.range.hard.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.hard.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.hard.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.hard.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.hard.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=1.0, range.mirror=F)
  )
}

for (operation in c('reciprocal')){
  cases = rbind(cases,
    c(parameter='zero.range.easy.nmru', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.easy.nmru', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.easy.nmru', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.easy.nmru', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.easy.nmru', operation=operation, simple=F, input.size=1, subset.ratio=1, overlap.ratio=0, range.a=0, range.b=1, range.mirror=F)
  )
}

for (operation in c('reciprocal')){
  cases = rbind(cases,
    c(parameter='zero.range.medium.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.0001, range.mirror=F),
    c(parameter='zero.range.medium.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.001, range.mirror=F),
    c(parameter='zero.range.medium.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.01, range.mirror=F),
    c(parameter='zero.range.medium.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=0.1, range.mirror=F),
    c(parameter='zero.range.medium.nmru', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0, range.b=1, range.mirror=F)
  )
}

eps = data.frame(rbind(
  c(operation='mul', epsilon=0.00001),
  c(operation='add', epsilon=0.00001),
  c(operation='sub', epsilon=0.00001),
  c(operation='div', epsilon=0.00001),
  c(operation='squared', epsilon=0.00001),
  c(operation='root', epsilon=0.00001),
  c(operation='reciprocal', epsilon=0.00001)
))

dat = data.frame(cases) %>%
  merge(eps) %>%
  mutate(
    simple=as.logical(as.character(simple)),
    input.size=as.integer(as.character(input.size)),
    subset.ratio=as.numeric(as.character(subset.ratio)),
    overlap.ratio=as.numeric(as.character(overlap.ratio)),
    range.a=as.numeric(as.character(range.a)),
    range.b=as.numeric(as.character(range.b)),
    #range.b=replace(range.b, range.b==as.numeric(1e-4),as.numeric(format(as.numeric(1e-4),scientific=FALSE))),
    range.mirror=as.logical(as.character(range.mirror)),
    epsilon=as.numeric(as.character(epsilon))
  ) %>%
  rowwise() %>%
  mutate(
    extrapolation.range=ifelse(range.mirror, ifelse(range.b<=0, paste0('U[',range.b*-1,',',range.a*-1,'] u U[',range.a,',',range.b,']') , paste0('U[-',range.b,',-',range.a,'] u U[',range.a,',',range.b,']')), paste0('U[',range.a,',',range.b,']')), # if a and b are -ve then want: [[+,+],[-,-]]
    extrapolation.range=gsub(",1e-04]",",0.0001]",extrapolation.range),  # for divBy0.range case as anything <=1e-04 is rep. in scientific notation automatically, and turning this off results in numbers getting additional zeros e.g. 0.1000
#    extrapolation.range=gsub(",1]",",1.0]",extrapolation.range),  # for divBy0.range case but is redundant as the range in python should use 1 not 1.0 
    operation=paste0('op-', operation)
  )
  
write.csv(dat, file="exp_setups.csv", row.names=F)

