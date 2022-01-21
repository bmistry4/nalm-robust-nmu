library(readr)
dat <- read_csv('/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_variations-G_clip.csv')
dat$name <- paste('G', dat$name, sep='')
write_csv(dat, '/data/bm4g15/nalu-stable-exp/csvs/SLTR_RealNPU_variations-G_clip_RENAMED.csv')
