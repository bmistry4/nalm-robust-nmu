
library(latex2exp)

#model.full.to.short = c(
#  'nac'='$\\mathrm{NAC}_{+}$',
#)
#
#model.latex.to.exp = c(
#  '$\\mathrm{NAC}_{+}$'=TeX('$\\mathrm{NAC}_{+}$')
#)
#
#model.to.exp = function(v) {
#  return(unname(revalue(v, model.latex.to.exp)))
#}

operation.full.to.short = c(
  'op-add'='$\\bm{+}$',
  'op-sub'='$\\bm{-}$',
  'op-mul'='$\\bm{\\times}$',
  'op-div'='$\\bm{\\mathbin{/}}$',
  'op-squared'='$z^2$',
  'op-root'='$\\sqrt{z}$'
)


extract.by.split = function (name, index, default=NA) {
  split = strsplit(as.character(name), '_')[[1]]
  if (length(split) >= index) {
    return(split[index])
  } else {
    return(default)
  }
}

expand.name = function (df) {
  names = data.frame(name=unique(df$name))

  df.expand.name = names %>%
    rowwise() %>%
    mutate(
      #model=revalue(extract.by.split(name, 1), model.full.to.short, warn_missing=FALSE),
      id = as.integer(extract.by.split(name, 1)),
      fold = as.integer(substring(extract.by.split(name, 2), 2)),
      operation=revalue(extract.by.split(name, 3), operation.full.to.short, warn_missing=FALSE)
    )

  df.expand.name$name = as.factor(df.expand.name$name)
  df.expand.name$id = as.factor(df.expand.name$id)
  df.expand.name$fold = as.factor(df.expand.name$fold)
  df.expand.name$operation = factor(df.expand.name$operation, c('$\\bm{\\times}$', '$\\bm{\\mathbin{/}}$', '$\\bm{+}$', '$\\bm{-}$', '$\\sqrt{z}$', '$z^2$'))

  #return(df.expand.name)
  return(merge(df, df.expand.name))
}
