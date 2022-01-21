import stable_nalu
import csv

fieldnames = ['seed', 'index.1', 'index.2']
csv_dict = {field: [] for field in fieldnames}
save_filename = 'dataset_solution_indexes.csv'


def update_csv_dict(seed, idx1, idx2):
    csv_dict['seed'].append(seed)
    csv_dict['index.1'].append(idx1)
    csv_dict['index.2'].append(idx2)


max_seeds = 25
for s in range(max_seeds):
    print(f"Seed: {s}")
    dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
        operation='add',
        input_size=10,
        subset_ratio=0.1,
        overlap_ratio=0,
        num_subsets=2,
        simple=False,
        use_cuda=False,
        seed=s,
    )

    print(f'  - dataset: {dataset.print_operation()}')
    update_csv_dict(s, idx1=dataset.subset_ranges[0][0], idx2=dataset.subset_ranges[1][0])

zd = zip(*csv_dict.values())
with open(save_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(csv_dict.keys())
    writer.writerows(zd)
print(f'csv file saved under name: {save_filename}')
