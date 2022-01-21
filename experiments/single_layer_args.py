import argparse
import ast

import torch

import stable_nalu


def parse_sample_distribution_args(parameter_info):
    allowed_distributions = ['uniform', 'truncated-normal', 'exponential', 'benford']
    if parameter_info[0] not in allowed_distributions:
        raise ValueError(f"Invalid sample-distribution family given. Allowed distributions:  {allowed_distributions}")
    # assumes first parameter is the distribution family (a string) and the rest of the parameters are floats
    for idx in range(1, len(parameter_info)):
        parameter_info[idx] = float(parameter_info[idx])
    return tuple(parameter_info)


def create_base_parser():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Runs the simple function static task')
    parser.add_argument('--id',
                        action='store',
                        default=-1,
                        type=int,
                        help='Unique id to identify experiment')
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
                            'add', 'sub', 'mul', 'div', 'squared', 'root', 'reciprocal'
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
                        default=[1, 2],
                        type=ast.literal_eval,
                        help='Specify the interpolation range that is sampled uniformly from')
    parser.add_argument('--extrapolation-range',
                        action='store',
                        default=[2, 6],
                        type=ast.literal_eval,
                        help='Specify the extrapolation range that is sampled uniformly from')
    parser.add_argument('--input-size',
                        action='store',
                        default=2,
                        type=int,
                        help='Specify the input size')
    parser.add_argument('--output-size',
                        action='store',
                        default=1,
                        type=int,
                        help='Specify the output size')
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

    parser.add_argument('--reg-scale-type',
                        action='store',
                        default='heim',
                        choices=['heim', 'madsen'],
                        type=str,
                        help='Type of npu regularisation scaling to use. Matches respective author\'s papers')
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
                        action='store',
                        default=0,
                        type=int,
                        help='Use sparisty reg on npu weights. Int represents the amount to scale reg by. 0 means off')
    parser.add_argument('--regualizer-gate',
                        type=int,
                        default=0,
                        help='Use sparisty reg on npu gate. Int represents the amount to scale reg by. 0 means off')
    parser.add_argument('--npu-clip',
                        action='store',
                        default='none',
                        choices=['none', 'w', 'g', 'wg', 'wig'],
                        help='Type of parameters (if any) to clip in a NPU/RealNPU module')
    parser.add_argument('--npu-Wr-init',
                        action='store',
                        default='xavier-uniform',
                        choices=['xavier-uniform', 'xavier-uniform-constrained'],
                        help='Init method to use for the W_real of the NPU. xavier-uniform= NPU paper init method,'
                             'xavier-uniform-constrained= NAU init method')

    parser.add_argument('--pytorch-precision',
                        type=int,
                        default=32,
                        help='Precision for pytorch to work in')

    parser.add_argument('--nmu-noise',
                        action='store_true',
                        default=False,
                        help='Applies/ unapplies multiplicative noise from a ~U[1,5] during training. Aids with failure ranges on a vinilla NMU.')
    parser.add_argument('--nau-noise',
                        action='store_true',
                        default=False,
                        help='Applies/ unapplies additive noise from a ~U[1,5] during training.')

    parser.add_argument('--no-save',
                        action='store_true',
                        default=False,
                        help='Do not save model at the end of training')
    parser.add_argument('--load-checkpoint',
                        action='store_true',
                        default=False,
                        help='Loads a saved checkpoint and resumes training')

    parser.add_argument('--pcc2mse-iteration',
                        action='store',
                        default=25000,
                        type=int,
                        help='Epoch to switch from pcc to mse loss for training')
    parser.add_argument('--train-criterion',
                        action='store',
                        default='mse',
                        choices=['mse', 'pcc', 'pcc-mse', 'mape'],
                        type=str,
                        help='Train criterion to optimise on. pcc is the stable version (with epsilon). '
                             'pcc-mse will use pcc until iteration args.pcc2mse-iteration and then switch to mse.')
    parser.add_argument('--beta-nau',
                        action='store_true',
                        default=False,
                        help='Have nau weights initialised using a beta distribution B(7,7)')

    parser.add_argument('--log-interval',
                        action='store',
                        default=1000,
                        type=int,
                        help='Log to tensorboard every X epochs.')

    parser.add_argument('--clip-grad-norm',
                        action='store',
                        default=None,
                        type=float,
                        help='Norm clip value for gradients.')

    parser.add_argument('--nru-div-mode',
                        action='store',
                        default='div',
                        choices=['div', 'div-sepSign'],
                        help='Division type for NRU. div calcs mag and sign in one go. div-sepSign calcs sign separately')
    parser.add_argument('--realnpu-reg-type',
                        action='store',
                        default='W',
                        choices=['W', 'bias'],
                        help='W penalises {-1,1}. bias penalises {-1,0,1}.')

    return parser


def robustness_experiments_args(parser):
    parser.add_argument('--sample-distribution',
                        action='store',
                        nargs='+',
                        default=['uniform'],
                        help='Distribution to sample from including any parameters. (uniform), '
                             '(truncated-normal, mean, std), (exponential, scale), (benford). Range info is covered by the'
                             'interpolation and extrapolation ranges.')
    parser.add_argument('--noise-range',
                        action='store',
                        default=[1, 5],
                        type=ast.literal_eval,
                        help='Range at which the noise for applying stochasticity is taken from. (Originally for sNMU.)')

def print_parser_configuration(args: argparse.Namespace):
    # Print configuration
    print(f'running')
    print(f'  - layer_type: {args.id}')
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
    print(f'  - output_size: {args.output_size}')
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
    print(f'  - reg_scale_type: {args.reg_scale_type}')
    print(f'  - regualizer_beta_start: {args.regualizer_beta_start}')
    print(f'  - regualizer_beta_end: {args.regualizer_beta_end}')
    print(f'  - regualizer_beta_step: {args.regualizer_beta_step}')
    print(f'  - regualizer_beta_growth: {args.regualizer_beta_growth}')
    print(f'  - regualizer_l1: {args.regualizer_l1}')
    print(f'  - regualizer-npu-w: {args.regualizer_npu_w}')
    print(f'  - regualizer-gate: {args.regualizer_gate}')
    print(f'  - npu-clip: {args.npu_clip}')
    print(f'  - npu-Wr-init: {args.npu_Wr_init}')
    print(f'  -')
    print(f'  - pytorch-precision: {torch.get_default_dtype()}')
    print(f'  -')
    print(f'  - nmu-noise: {args.nmu_noise}')
    print(f'  - nau-noise: {args.nau_noise}')
    print(f'  -')
    print(f'  - no-save: {args.no_save}')
    print(f'  - load-checkpoint: {args.load_checkpoint}')
    print(f'  - log-interval: {args.log_interval}')
    print(f'  -')
    print(f'  - train-criterion: {args.train_criterion}')
    print(f'  - pcc2mse-iteration: {args.pcc2mse_iteration}')
    print(f'  -')
    print(f'  - beta-nau: {args.beta_nau}')
    print(f'  - clip-grad-norm: {args.clip_grad_norm}')
    print(f'  - nru_div_mode: {args.nru_div_mode}')
    print(f'  - realnpu_reg_type: {args.realnpu_reg_type}')
    print(f'  -')


def print_robustness_params(args):
    print(f'  - sample_distribution: {args.sample_distribution}')
    print(f'  - noise_range: {args.noise_range}')
    print(f'  -')

