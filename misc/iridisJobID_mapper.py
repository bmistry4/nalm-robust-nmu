import csv

# given an array id used for batch jobs on iridis, get the config (seed and range) related to it. Or visa-versa.

array_id = 0
interpolation_ranges = ['[-20,-10]', '[-2,-1]', '[-1.2,-1.1]', '[-0.2,-0.1]', '[-2,2]', '       [0.1,0.2]', '[1,2]',
                        '[1.1,1.2]', '[10,20]']
extrapolation_ranges = ['[-40,-20]', '[-6,-2]', '[-6.1,-1.2]', '[-2,-0.2]', '[[-6,-2],[2,6]]', '[0.2,2]', '[2,6]',
                        '[1.2,6]', '[20,40]']


def get_config_info(array_id, ranges: tuple):
    interp_ranges, extrap_ranges = ranges
    num_ranges = len(interp_ranges)
    seed = array_id // num_ranges
    range_idx = array_id % num_ranges if num_ranges > 1 else 0

    print(f'Array id: {array_id}, seed: {seed}, ranges: interpolation: {interp_ranges[range_idx]}; '
          f'extrapolation: {extrap_ranges[range_idx]}')


def get_array_id(seed, interp_range, all_interp_ranges):
    array_block = len(all_interp_ranges) * seed
    range_idx = all_interp_ranges.index(interp_range)
    offset = 0 if len(all_interp_ranges) < 2 else range_idx
    array_id = array_block + offset
    print(f'Array id: {array_id}, seed: {seed}, ranges: interpolation: {all_interp_ranges[range_idx]}')


def parameter_to_range(parameter):
    # e.g. U[-2,1] to [-2,1]
    return parameter[1:]


def get_failure_runs_array_ids(csv_filepath):
    # get the array id given csv containing the ranges and seed the model fails on
    # to create csv run in R: `filter(dat, solved == FALSE) %>% select(parameter, seed)` on the 'best_seeds' csv
    with open(csv_filepath) as file:
        reader = csv.DictReader(file)
        for row in reader:
            get_array_id(int(row['seed']), parameter_to_range(row['parameter']), interpolation_ranges)


""" ID -> CONFIG INFO """
# get_config_info(array_id=209, ranges=(interpolation_ranges, extrapolation_ranges))
""" CONFIG INFO -> ARRAY ID FOR 25 seeds """
for r in interpolation_ranges:
    for s in range(25):
        get_array_id(seed=s, interp_range=r, all_interp_ranges=interpolation_ranges)
""" FAILURE RUNS -> ARRAY IDS given a csv file """
# failures_csv = r'WG_clip_failures.csv'
# get_failure_runs_array_ids(failures_csv)

