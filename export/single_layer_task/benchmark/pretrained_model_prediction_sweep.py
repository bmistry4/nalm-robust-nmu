# load in pretrained model on 2 input arithmetic task
# create data which grid samples between a upper and lower bound in equal steps
# get output
# save the in1, in2 and out data to csv
import ast
import os
import os.path as path
import argparse
import torch
import stable_nalu
import pandas as pd

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
                    choices=['normal', 'safe', 'trig', 'max-safe', 'mnac', 'golden-ratio'],
                    help='Multplication unit, can be normal, safe, trig')
parser.add_argument('--nalu-gate',
                    action='store',
                    default='normal',
                    choices=['normal', 'regualized', 'obs-gumbel', 'gumbel', 'golden-ratio'],
                    type=str,
                    help='Can be normal, regualized, obs-gumbel, or gumbel')
parser.add_argument('--nac-weight',
                    action='store',
                    default='normal',
                    choices=['normal', 'golden-ratio'],
                    type=str,
                    help='Way to calculate the NAC+.')

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

parser.add_argument('--noise-range',
                    action='store',
                    default=[1, 5],
                    type=ast.literal_eval,
                    help='Range at which the noise for applying stochasticity is taken from. (Originally for sNMU.)')

parser.add_argument('--clip-grad-value',
                    action='store',
                    default=None,
                    type=float,
                    help='Clip value for gradients i.e. [-value, value].')
parser.add_argument('--reinit',
                    action='store_true',
                    default=False,
                    help='Enables iNALU\'s reinitialization scheme')
parser.add_argument('--reinit-epoch-interval',
                    action='store',
                    default=10,
                    type=int,
                    help='Check after this many epochs if reinitialization can occur.')
parser.add_argument('--reinit-max-stored-losses',
                    action='store',
                    default=5000,
                    type=int,
                    help='Number of losses that need to be collected before reinitialization can occur.')
parser.add_argument('--reinit-loss-thr',
                    action='store',
                    default=1.,
                    type=float,
                    help='Reinitialization only occurs if the avg accumulated loss is greater than this threshold.')

parser.add_argument('--mlp-bias',
                    action='store_true',
                    default=False,
                    help='Enables bias in the MLP layers')
parser.add_argument('--mlp-depth',
                    action='store',
                    default=0,
                    type=int,
                    help='Number of hidden layers in the MLP (i.e. depth).')

##############################################################
# Parser args specific to this task
parser.add_argument('--model-filepath',
                    type=str,
                    help='Filepath to where all the model is saved')
parser.add_argument('--csv-save-folder',
                    type=str,
                    help='Folder where the csv results will be saved')
parser.add_argument('--csv-save-filename',
                    type=str,
                    help='Filename for the csv results')
parser.add_argument('--lower-bound',
                    type=int,
                    default=-6,
                    help='Lower bound to sample the two inputs from.')
parser.add_argument('--upper-bound',
                    type=int,
                    default=6,
                    help='Upper bound to sample the two inputs from.')
parser.add_argument('--step-size',
                    type=float,
                    default=0.1,
                    help='Step size to sample grid points.')
parser.add_argument('--gold-outputs',
                    action='store_true',
                    default=False,
                    help='If to generate results for the golden values (i.e. true output)')
##############################################################

args = parser.parse_args()

# generate input data (i.e. grid)
x1 = torch.arange(args.lower_bound, args.upper_bound + args.step_size, args.step_size)
x2 = torch.arange(args.lower_bound, args.upper_bound + args.step_size, args.step_size)
input_data = torch.cartesian_prod(x1, x2)

# setup model
model = stable_nalu.network.SingleLayerNetwork(
    args.layer_type,
    input_size=input_data.shape[1],
    output_size=args.output_size,
    writer=stable_nalu.writer.DummySummaryWriter().every(args.log_interval).verbose(args.verbose),
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
    nac_weight=args.nac_weight,
    regualizer_gate=args.regualizer_gate,
    regualizer_npu_w=args.regualizer_npu_w,
    nmu_noise=args.nmu_noise,
    nau_noise=args.nau_noise,
    beta_nau=args.beta_nau,
    npu_clip=args.npu_clip,
    npu_Wr_init=args.npu_Wr_init,
    nru_div_mode=args.nru_div_mode,
    realnpu_reg_type=args.realnpu_reg_type,
    noise_range=args.noise_range,
    bias=args.mlp_bias,
    mlp_depth=args.mlp_depth
)

# generate gold solutions for the different operations
if args.gold_outputs:
    # multiplication
    gold_mul_data_to_save = torch.cat((input_data, torch.prod(input_data, dim=1).unsqueeze(1)), dim=1)
    df = pd.DataFrame(gold_mul_data_to_save.detach().numpy(), columns=['x1', 'x2', 'pred'])
    df.to_csv(path.join(args.csv_save_folder, 'mul_gold') + '.csv', index=False)
    
    # division
    gold_div_data_to_save = torch.cat((input_data, (input_data[:,0] / input_data[:,1]).unsqueeze(1)), dim=1)
    df = pd.DataFrame(gold_div_data_to_save.detach().numpy(), columns=['x1', 'x2', 'pred'])
    df.to_csv(path.join(args.csv_save_folder, 'div_gold') + '.csv', index=False)

    # addition
    gold_add_data_to_save = torch.cat((input_data, torch.sum(input_data, dim=1).unsqueeze(1)), dim=1)
    df = pd.DataFrame(gold_add_data_to_save.detach().numpy(), columns=['x1', 'x2', 'pred'])
    df.to_csv(path.join(args.csv_save_folder, 'add_gold') + '.csv', index=False)
    
    # subtraction
    gold_sub_data_to_save = torch.cat((input_data, (input_data[:,0] - input_data[:,1]).unsqueeze(1)), dim=1)
    df = pd.DataFrame(gold_sub_data_to_save.detach().numpy(), columns=['x1', 'x2', 'pred'])
    df.to_csv(path.join(args.csv_save_folder, 'sub_gold') + '.csv', index=False)

else:
    load_file = args.model_filepath
    csv_filepath = path.join(args.csv_save_folder, args.csv_save_filename) + '.csv'

    # load model
    if not os.path.isfile(load_file):
        raise FileExistsError(f'Model save file: {load_file} (to load checkpoint) doesn\'t exist')
    checkpoint = torch.load(load_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    # predict output values using pretrianed model
    model.eval()
    output = model(input_data)
    print('output shape:', output.shape)

    # save inputs and predictions
    data_to_save = torch.cat((input_data, output), dim=1)
    df = pd.DataFrame(data_to_save.detach().numpy(), columns=['x1', 'x2', 'pred'])
    df.to_csv(csv_filepath, index=False)

print("Script completed.")
