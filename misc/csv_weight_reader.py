import pandas as pd

#python3 csv_weight_reader.py # (assuming you're on iridis)

# will read tb converted csv file and operate on its dataframe to analyse weight values
# designed for the 1 layer task

read_filepath = "/data/bm4g15/nalu-stable-exp/csvs/r_results/single_layer_task/NMU_noise_failures/i-1.1-1.2_noise-1-5_V.csv"  
save_filepath = "/data/bm4g15/nalu-stable-exp/csvs/r_results/single_layer_task/NMU_noise_failures/i-1.1-1.2_noise-1-5_V_E0_stats.csv"
df = pd.read_csv(read_filepath)
# get weights
df = df[df.step==0].sort_values('name')
# calc mean and variance
df_weight_cols = df[['weights.w0','weights.w1']]
df['weights.mean'] = df_weight_cols.mean(1)
df['weights.var'] = df_weight_cols.var(1, ddof=1)
# calc no. positive weights 
df['weights.pos.count']=(df['weights.w0']> 0).astype(int) +  (df['weights.w1']> 0).astype(int)
df = df.filter(["name", "seed", "metric.valid.interpolation", "step", "metric.test.extrapolation", "weights.w0", "weights.w1", "sparse.error.max", "weights.mean", "weights.var", "weights.pos.count"])
#save df as csv
df.to_csv(save_filepath, index=False)

