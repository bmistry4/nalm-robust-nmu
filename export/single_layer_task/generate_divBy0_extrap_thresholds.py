"""
This file contains the parameter configurations to calculate the acceptable MSE extrpolation threshold to be considered
a success in the three different divBy0 tasks (easy, medium, and hard).
- For each, we manually set the weight values to the 'gold' solution and calculate the resulting extrapolation loss.
- we will also add a +eps to the loss to avoid further issues with precision.
"""
import stable_nalu
import torch
import math
import csv
from decimal import Decimal

torch.set_default_dtype(torch.float32)
seed = 0  # seed values doesn't matter for this
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
eps = 0#torch.finfo().eps  #TODO - Use 0 when generating golden thresholds and use finfo() when generating threshold vals

criterion = torch.nn.MSELoss()
ranges = [[0, 1e-8], [0, 1e-7], [0, 1e-6], [0, 1e-5], [0, 1e-4], [0, 1e-3], [0, 1e-2], [0, 1e-1], [0, 1]]

fieldnames = ['module', 'parameter', 'test.mse', 'range']
csv_dict = {field: [] for field in fieldnames}
range_to_str = {r[1]: f"U[0,{Decimal(r[1]):.0e})" for r in ranges}

"""
Easy - in:[a] out:1/a -> Gold solution: self.W = torch.nn.Parameter(torch.Tensor([[-1.]]))

Same as running the following (assuming W has been fixed)
python3 -u single_layer.py \
--operation reciprocal --layer-type NRU --nac-mul mnac --input-size 1 --subset-ratio 1 --num-subsets 1
--max-iterations 1 --remove-existing-data --no-cuda --no-save --pytorch-precision 32
"""
easy_dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation='reciprocal',
    input_size=1,
    subset_ratio=1,
    overlap_ratio=0,
    num_subsets=1,
    simple=False,
    use_cuda=False,
    seed=seed,
)

"""
Medium - in:[a,b] out:1/a -> Gold solution: self.W = torch.nn.Parameter(torch.Tensor([[-1.,0.]]))

Same as running the following (assuming W has been fixed)
python3 -u single_layer.py \
--operation reciprocal --layer-type NRU --nac-mul mnac --max-iterations 1 --remove-existing-data --no-cuda --no-save
--pytorch-precision 32
"""
medium_dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation='reciprocal',
    input_size=2,
    subset_ratio=0.5,
    overlap_ratio=0,
    num_subsets=2,
    simple=False,
    use_cuda=False,
    seed=seed,
)

"""
Hard: - in:[a,b] out:a/b -> Gold solution: self.W = torch.nn.Parameter(torch.Tensor([[1., -1.]]))

Same as running the following (assuming W has been fixed)
python3 -u single_layer.py \
--operation div --layer-type NRU --nac-mul mnac --max-iterations 1 --remove-existing-data --no-cuda --no-save
--pytorch-precision 32
"""
hard_dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation='div',
    input_size=2,
    subset_ratio=0.5,
    overlap_ratio=0,
    num_subsets=2,
    simple=False,
    use_cuda=False,
    seed=seed,
)


def update_csv_dict(module, param, error, range):
    csv_dict['module'].append(module)
    csv_dict['parameter'].append(param)
    csv_dict['test.mse'].append(error)
    csv_dict['range'].append(range_to_str[range[1]])


"""
NRU
"""
print("*" * 20)
print("NEURAL RECIPROCAL UNIT (NRU)")
print("*" * 20)
model = 'NRU'
print(f'  - easy_dataset: {easy_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[-1.]]))
for range in ranges:
    easy_dataset_extrapolation = next(
        iter(easy_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = easy_dataset_extrapolation
    pred = (x[:, 0] ** W[0][0]).unsqueeze(1)  # manually calc division
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'easy', thr_err.item(), range)
    print(f'{range}:{thr_err.item()}')

print()
print(f'  - medium_dataset: {medium_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[-1., 0.]]))
for range in ranges:
    dataset_extrapolation = next(
        iter(medium_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = dataset_extrapolation
    pred = ((x[:, 0] ** W[0][0]) * (x[:, 1] ** W[0][1])).unsqueeze(1)  # manually calc division
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'medium', thr_err.item(), range)
    print(f'{range}:{thr_err.item()}')

print()
print(f'  - hard_dataset: {hard_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[1., -1.]]))
for range in ranges:
    dataset_extrapolation = next(
        iter(hard_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = dataset_extrapolation
    pred = ((x[:, 0] ** W[0][0]) * (x[:, 1] ** W[0][1])).unsqueeze(1)  # manually calc division
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'hard', thr_err.item(), range)
    print(f'{range}:{thr_err.item()}')

print()
"""
RealNPU (with eps)
"""
print("*" * 20)
print("Real NPU")
print("*" * 20)
print(f'  - easy_dataset: {easy_dataset.print_operation()}')
model = 'RealNPU'
npu_eps = torch.finfo().eps

def realnpu_pred(X, gold_W, npu_eps):
    r = (torch.abs(X) + npu_eps) * g + (1 - g)
    k = (torch.max(-torch.sign(X), torch.zeros_like(X)) * math.pi) * g
    pred = torch.exp(torch.log(r).matmul(gold_W)) * torch.cos(k.matmul(gold_W))
    return pred

W = torch.nn.Parameter(torch.Tensor([[-1.]]))
g = torch.nn.Parameter(torch.Tensor([1.]))
for range in ranges:
    easy_dataset_extrapolation = next(
        iter(easy_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = easy_dataset_extrapolation
    pred = realnpu_pred(x, W, npu_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'easy', thr_err.item(), range)
    print(f'{range}: {thr_err.item()}')
print()

print(f'  - medium_dataset: {medium_dataset.print_operation()}')
# Alternate solution: W = [-1,0] and g=[1,1]
W = torch.nn.Parameter(torch.Tensor([[-1., 1.]])).T
g = torch.nn.Parameter(torch.Tensor([1., 0.]))
for range in ranges:
    medium_dataset_extrapolation = next(
        iter(medium_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = medium_dataset_extrapolation
    pred = realnpu_pred(x, W, npu_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'medium', thr_err.item(), range)
    print(f'{range}: {thr_err.item()}')
print()

print(f'  - hard_dataset: {hard_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[1., -1.]])).T
g = torch.nn.Parameter(torch.Tensor([1., 1.]))
for range in ranges:
    hard_dataset_extrapolation = next(
        iter(hard_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = hard_dataset_extrapolation
    pred = realnpu_pred(x, W, npu_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'hard', thr_err.item(), range)
    print(f'{range}: {thr_err.item()}')

"""
RealNPU (without eps)
"""
print("*" * 20)
print("Real NPU (without eps in model)")
print("*" * 20)
print(f'  - easy_dataset: {easy_dataset.print_operation()}')
model = 'RealNPU (eps=0)'
npu_eps = 0

W = torch.nn.Parameter(torch.Tensor([[-1.]]))
g = torch.nn.Parameter(torch.Tensor([1.]))
for range in ranges:
    easy_dataset_extrapolation = next(
        iter(easy_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = easy_dataset_extrapolation
    pred = realnpu_pred(x, W, npu_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'easy', thr_err.item(), range)
    print(f'{range}: {thr_err.item()}')
print()

print(f'  - medium_dataset: {medium_dataset.print_operation()}')
# Alternate solution: W = [-1,0] and g=[1,1]
W = torch.nn.Parameter(torch.Tensor([[-1., 1.]])).T
g = torch.nn.Parameter(torch.Tensor([1., 0.]))
for range in ranges:
    medium_dataset_extrapolation = next(
        iter(medium_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = medium_dataset_extrapolation
    pred = realnpu_pred(x, W, npu_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'medium', thr_err.item(), range)
    print(f'{range}: {thr_err.item()}')
print()

print(f'  - hard_dataset: {hard_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[1., -1.]])).T
g = torch.nn.Parameter(torch.Tensor([1., 1.]))
for range in ranges:
    hard_dataset_extrapolation = next(
        iter(hard_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = hard_dataset_extrapolation
    pred = realnpu_pred(x, W, npu_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'hard', thr_err.item(), range)
    print(f'{range}: {thr_err.item()}')

"""
NMRU
"""
print("*" * 20)
print("NEURAL MULTIPLICATION RECIPROCAL UNIT (NMRU)")
print("*" * 20)
model = 'NMRU'
nmru_eps = 0  # only add eps when training


def nmru_pred(X, gold_W, nmru_eps):
    reciprocal = X.reciprocal() + nmru_eps
    X = torch.cat((X, reciprocal), 1)  # concat on input dim
    k = torch.max(-torch.sign(X), torch.zeros_like(X)) * math.pi
    X = torch.abs(X)

    out_size, in_size = gold_W.size()
    mnac_x = X.view(X.size()[0], in_size, 1)
    mnac_W = gold_W.t().view(1, in_size, out_size)
    magnitude = torch.prod(mnac_x * mnac_W + 1 - mnac_W, -2)

    pred = magnitude * torch.cos(k.matmul(gold_W.T))
    return pred

print(f'  - easy_dataset: {easy_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[0., 1.]]))
for range in ranges:
    easy_dataset_extrapolation = next(
        iter(easy_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = easy_dataset_extrapolation
    pred = nmru_pred(x, W, nmru_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'easy', thr_err.item(), range)
    print(f'{range}:{thr_err.item()}')

print()
print(f'  - medium_dataset: {medium_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[0., 0., 1., 0.]]))
for range in ranges:
    dataset_extrapolation = next(
        iter(medium_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = dataset_extrapolation
    pred = nmru_pred(x, W, nmru_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'medium', thr_err.item(), range)
    print(f'{range}:{thr_err.item()}')

print()
print(f'  - hard_dataset: {hard_dataset.print_operation()}')
W = torch.nn.Parameter(torch.Tensor([[1., 0., 0., 1.]]))
for range in ranges:
    dataset_extrapolation = next(
        iter(hard_dataset.fork(sample_range=range, seed=8689336).dataloader(batch_size=10000)))
    x, y = dataset_extrapolation
    pred = nmru_pred(x, W, nmru_eps)
    err = criterion(pred, y)
    thr_err = err + eps
    update_csv_dict(model, 'hard', thr_err.item(), range)
    print(f'{range}:{thr_err.item()}')

print()

"""
WRITE CSV FILE 
To be used in generating the plot comparing gold test errors for the easy, medium, and hard tasks 
Make sure eps=0
"""
assert eps == 0, f"eps value must be 0 for writing to csv. Currently, eps={eps}"
zd = zip(*csv_dict.values())
with open('divBy0_test_errors.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(csv_dict.keys())
    writer.writerows(zd)
print('csv file saved')
