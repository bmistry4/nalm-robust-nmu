import os
import ast
import math
import torch
import stable_nalu
import argparse
import stable_nalu.functional.regualizer as Regualizer
from decimal import Decimal

# Prints out the weight inits for the model. (Will not do tb logging/ run model). No data is created.

# python_lfs_job-iridis.sh misc/single_layer_weight_init_check.py     --operation sub --layer-type ReRegualizedLinearNAC     --regualizer-scaling-start 5000 --regualizer-scaling-end 25000 --regualizer 0.01     --interpolation-range '[1.1,1.2]' --extrapolation-range '[1.2,6]'     --seed 0 --max-iterations 1     --name-prefix test --remove-existing-data --no-cuda
#  python_lfs_job-iridis.sh misc/single_layer_weight_init_check.py     --operation sub --layer-type ReRegualizedLinearNAC     --regualizer-scaling-start 5000 --regualizer-scaling-end 25000 --regualizer 0.01     --interpolation-range '[-1.2,-1.1]' --extrapolation-range '[-6.1,-1.2]'     --seed 0 --max-iterations 1     --name-prefix test --remove-existing-data --no-cuda
# Parse arguments

parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SimpleFunctionStaticNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. Tanh, ReLU, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=[
                        'add', 'sub', 'mul', 'div', 'squared', 'root'
                    ],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul, squared')
parser.add_argument('--num-subsets',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the number of subsets to use')
parser.add_argument('--regualizer',
                    action='store',
                    default=10,
                    type=float,
                    help='Specify the regualization lambda to be used')
parser.add_argument('--regualizer-z',
                    action='store',
                    default=0,
                    type=float,
                    help='Specify the z-regualization lambda to be used')
parser.add_argument('--regualizer-oob',
                    action='store',
                    default=1,
                    type=float,
                    help='Specify the oob-regualization lambda to be used')
parser.add_argument('--first-layer',
                    action='store',
                    default=None,
                    help='Set the first layer to be a different type')

parser.add_argument('--max-iterations',
                    action='store',
                    default=100000,
                    type=int,
                    help='Specify the max number of iterations to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=128,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')

parser.add_argument('--interpolation-range',
                    action='store',
                    default=[1,2],
                    type=ast.literal_eval,
                    help='Specify the interpolation range that is sampled uniformly from')
parser.add_argument('--extrapolation-range',
                    action='store',
                    default=[2,6],
                    type=ast.literal_eval,
                    help='Specify the extrapolation range that is sampled uniformly from')
parser.add_argument('--input-size',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the input size')
parser.add_argument('--subset-ratio',
                    action='store',
                    default=0.5,
                    type=float,
                    help='Specify the subset-size as a fraction of the input-size')
parser.add_argument('--overlap-ratio',
                    action='store',
                    default=0.0,
                    type=float,
                    help='Specify the overlap-size as a fraction of the input-size')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])')

parser.add_argument('--hidden-size',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the vector size of the hidden layer.')
parser.add_argument('--nac-mul',
                    action='store',
                    default='none',
                    choices=['none', 'normal', 'safe', 'max-safe', 'mnac', 'npu', 'real-npu'],
                    type=str,
                    help='Make the second NAC a multiplicative NAC, used in case of a just NAC network.')
parser.add_argument('--oob-mode',
                    action='store',
                    default='clip',
                    choices=['regualized', 'clip'],
                    type=str,
                    help='Choose of out-of-bound should be handled by clipping or regualization.')
parser.add_argument('--regualizer-scaling',
                    action='store',
                    default='linear',
                    choices=['exp', 'linear'],
                    type=str,
                    help='Use an expoentational scaling from 0 to 1, or a linear scaling.')
parser.add_argument('--regualizer-scaling-start',
                    action='store',
                    default=1000000,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=2000000,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer-shape',
                    action='store',
                    default='linear',
                    choices=['squared', 'linear', 'none'],
                    type=str,
                    help='Use either a squared or linear shape for the bias and oob regualizer. Use none so W reg in tensorboard is logged at 0')
parser.add_argument('--mnac-epsilon',
                    action='store',
                    default=0,
                    type=float,
                    help='Set the idendity epsilon for MNAC.')
parser.add_argument('--nalu-bias',
                    action='store_true',
                    default=False,
                    help='Enables bias in the NALU gate')
parser.add_argument('--nalu-two-nac',
                    action='store_true',
                    default=False,
                    help='Uses two independent NACs in the NALU Layer')
parser.add_argument('--nalu-two-gate',
                    action='store_true',
                    default=False,
                    help='Uses two independent gates in the NALU Layer')
parser.add_argument('--nalu-mul',
                    action='store',
                    default='normal',
                    choices=['normal', 'safe', 'trig', 'max-safe', 'mnac'],
                    help='Multplication unit, can be normal, safe, trig')
parser.add_argument('--nalu-gate',
                    action='store',
                    default='normal',
                    choices=['normal', 'regualized', 'obs-gumbel', 'gumbel'],
                    type=str,
                    help='Can be normal, regualized, obs-gumbel, or gumbel')

parser.add_argument('--optimizer',
                    action='store',
                    default='adam',
                    choices=['adam', 'sgd'],
                    type=str,
                    help='The optimization algorithm to use, Adam or SGD')
parser.add_argument('--learning-rate',
                    action='store',
                    default=1e-3,
                    type=float,
                    help='Specify the learning-rate')
parser.add_argument('--momentum',
                    action='store',
                    default=0.0,
                    type=float,
                    help='Specify the nestrov momentum, only used with SGD')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='simple_function_static',
                    type=str,
                    help='Where the data should be stored')
parser.add_argument('--remove-existing-data',
                    action='store_true',
                    default=False,
                    help='Should old results be removed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')

parser.add_argument('--regualizer-beta-start',
                    action='store',
                    default=1e-5,
                    type=float,
                    help='Starting value of the beta scale factor.')
parser.add_argument('--regualizer-beta-end',
                    action='store',
                    default=1e-4,
                    type=float,
                    help='Final value of the beta scale factor.')
parser.add_argument('--regualizer-beta-step',
                    action='store',
                    default=10000,
                    type=int,
                    help='Update the regualizer-beta-start value every x steps.')
parser.add_argument('--regualizer-beta-growth',
                    action='store',
                    default=10,
                    type=int,
                    help='Scale factor to grow the regualizer-beta-start value by.')
parser.add_argument('--regualizer-l1',
                    action='store_true',
                    default=False,
                    help='Add L1 regularization loss term. Be sure the regualizer-scaling is set')
parser.add_argument('--regualizer-npu-w',
                    action='store_true',
                    default=False,
                    help='Add L1 regularization loss term. Be sure the regualizer-scaling is set')
parser.add_argument('--regualizer-gate',
                    type=int,
                    default=0,
                    help='')

parser.add_argument('--pytorch-precision',
                    type=int,
                    default=32,
                    help='Precision for pytorch to work in')
                    
                    
def print_model_params(model):
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      #print('{0:.6f}'.format(param.data[0][0].item()))
      print(param.data)
  print()

args = parser.parse_args()

if args.pytorch_precision == 32:
  torch.set_default_dtype(torch.float32)
elif args.pytorch_precision == 64:
  torch.set_default_dtype(torch.float64)
else:
  raise ValueError(f'Unsupported pytorch_precision option ({args.pytorch_precision})')

setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)
torch.set_printoptions(precision=6)

# Print configuration
print(f'running')
print(f'  - layer_type: {args.layer_type}')
print(f'  - first_layer: {args.first_layer}')
print(f'  - operation: {args.operation}')
print(f'  - num_subsets: {args.num_subsets}')
print(f'  - regualizer: {args.regualizer}')
print(f'  - regualizer_z: {args.regualizer_z}')
print(f'  - regualizer_oob: {args.regualizer_oob}')
print(f'  -')
print(f'  - max_iterations: {args.max_iterations}')
print(f'  - batch_size: {args.batch_size}')
print(f'  - seed: {args.seed}')
print(f'  -')
print(f'  - interpolation_range: {args.interpolation_range}')
print(f'  - extrapolation_range: {args.extrapolation_range}')
print(f'  - input_size: {args.input_size}')
print(f'  - subset_ratio: {args.subset_ratio}')
print(f'  - overlap_ratio: {args.overlap_ratio}')
print(f'  - simple: {args.simple}')
print(f'  -')
print(f'  - hidden_size: {args.hidden_size}')
print(f'  - nac_mul: {args.nac_mul}')
print(f'  - oob_mode: {args.oob_mode}')
print(f'  - regualizer_scaling: {args.regualizer_scaling}')
print(f'  - regualizer_scaling_start: {args.regualizer_scaling_start}')
print(f'  - regualizer_scaling_end: {args.regualizer_scaling_end}')
print(f'  - regualizer_shape: {args.regualizer_shape}')
print(f'  - mnac_epsilon: {args.mnac_epsilon}')
print(f'  - nalu_bias: {args.nalu_bias}')
print(f'  - nalu_two_nac: {args.nalu_two_nac}')
print(f'  - nalu_two_gate: {args.nalu_two_gate}')
print(f'  - nalu_mul: {args.nalu_mul}')
print(f'  - nalu_gate: {args.nalu_gate}')
print(f'  -')
print(f'  - optimizer: {args.optimizer}')
print(f'  - learning_rate: {args.learning_rate}')
print(f'  - momentum: {args.momentum}')
print(f'  -')
print(f'  - cuda: {args.cuda}')
print(f'  - name_prefix: {args.name_prefix}')
print(f'  - remove_existing_data: {args.remove_existing_data}')
print(f'  - verbose: {args.verbose}')
print(f'  -')
print(f'  - regualizer_beta_start: {args.regualizer_beta_start}')
print(f'  - regualizer_beta_end: {args.regualizer_beta_end}')
print(f'  - regualizer_beta_step: {args.regualizer_beta_step}')
print(f'  - regualizer_beta_growth: {args.regualizer_beta_growth}')
print(f'  - regualizer_l1: {args.regualizer_l1}')
print(f'  - regualizer-npu-w: {args.regualizer_npu_w}')
print(f'  - regualizer-gate: {args.regualizer_gate}')
print(f'  - pytorch-precision: {torch.get_default_dtype()}')

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    torch.set_num_threads(int(os.environ['LSB_DJOB_NUMPROC']))

def init():
    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Setup datasets
    dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
        operation=args.operation,
        input_size=args.input_size,
        subset_ratio=args.subset_ratio,
        overlap_ratio=args.overlap_ratio,
        num_subsets=args.num_subsets,
        simple=args.simple,
        use_cuda=args.cuda,
        seed=args.seed,
    )

    # setup model
    model = stable_nalu.network.SingleLayerNetwork(
        args.layer_type,
        input_size=dataset.get_input_size(),
        #writer=summary_writer.every(1000).verbose(args.verbose),
        first_layer=args.first_layer,
        hidden_size=args.hidden_size,
        nac_oob=args.oob_mode,
        regualizer_shape=args.regualizer_shape,
        regualizer_z=args.regualizer_z,
        mnac_epsilon=args.mnac_epsilon,
        nac_mul=args.nac_mul,
        nalu_bias=args.nalu_bias,
        nalu_two_nac=args.nalu_two_nac,
        nalu_two_gate=args.nalu_two_gate,
        nalu_mul=args.nalu_mul,
        nalu_gate=args.nalu_gate,
        fixed_gate=False,       # TODO - create arg flag?
        regualizer_gate=args.regualizer_gate,
        regualizer_npu_w=args.regualizer_npu_w,
    )
    model.reset_parameters()

    print(f'seed: {args.seed}')
    print_model_params(model)

for s in range(15):
    args.seed = s
    init()
    

