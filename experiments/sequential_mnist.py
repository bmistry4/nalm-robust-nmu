
import os
import ast
import math
import torch
import stable_nalu
import argparse
import misc.utils as utils
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Run either the MNIST counting or MNIST Arithmetic task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SequentialMnistNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. RNN-tanh, LSTM, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='cumsum',
                    choices=[
                        'cumsum', 'sum', 'cumprod', 'prod', 'cumdiv', 'div'
                    ],
                    type=str,
                    help='Specify the operation to use, sum or count')
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
parser.add_argument('--mnist-digits',
                    action='store',
                    default=[0,1,2,3,4,5,6,7,8,9],
                    type=lambda str: list(map(int,str)),
                    help='MNIST digits to use')
parser.add_argument('--mnist-outputs',
                    action='store',
                    default=1,
                    type=int,
                    help='number of MNIST to use, more than 1 adds redundant values')
parser.add_argument('--model-simplification',
                    action='store',
                    default='none',
                    choices=[
                        'none', 'solved-accumulator', 'pass-through'
                    ],
                    type=str,
                    help='Simplifiations applied to the model')

parser.add_argument('--max-epochs',
                    action='store',
                    default=1000,
                    type=int,
                    help='Specify the max number of epochs to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=64,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')

parser.add_argument('--interpolation-length',
                    action='store',
                    default=10,
                    type=int,
                    help='Specify the sequence length for interpolation')
parser.add_argument('--extrapolation-lengths',
                    action='store',
                    default=[100, 1000],
                    type=ast.literal_eval,
                    help='Specify the sequence lengths used for the extrapolation dataset')

parser.add_argument('--softmax-transform',
                    action='store_true',
                    default=False,
                    help='Should a softmax transformation be used to control the output of the CNN model')
parser.add_argument('--nac-mul',
                    action='store',
                    default='none',
                    choices=['none', 'normal', 'safe', 'max-safe', 'mnac'],
                    type=str,
                    help='Make the second NAC a multiplicative NAC, used in case of a just NAC network.')
parser.add_argument('--nac-oob',
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
                    default=10000,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=100000,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer-shape',
                    action='store',
                    default='linear',
                    choices=['squared', 'linear'],
                    type=str,
                    help='Use either a squared or linear shape for the bias and oob regualizer.')
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

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='sequence_mnist',
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

parser.add_argument('--dataset-workers',
                    action='store',
                    default=1,
                    type=int,
                    help='Number of workers for multi-process data loading')

parser.add_argument('--nmu-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies multiplicative noise from a ~U[1,5] during training. Aids with failure ranges on a vinilla NMU.')
parser.add_argument('--nau-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies additive noise from a ~U[1,5] during training.')
parser.add_argument('--noise-range',
                    action='store',
                    default=[1, 5],
                    type=ast.literal_eval,
                    help='Range at which the noise for applying stochasticity is taken from. (Originally for sNMU.)')

parser.add_argument('--pcc2mse-iteration',
                    action='store',
                    default=7500,
                    type=int,
                    help='Iteration (global step) at which to switch from a pcc loss to a mse loss for training')
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

parser.add_argument('--load-checkpoint',
                    action='store_true',
                    default=False,
                    help='Loads a saved checkpoint and resumes training')


# if __name__ == '__main__':
args = parser.parse_args()

setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)

# print parser args
print(' '.join(f'{k}: {v}\n' for k, v in vars(args).items()))


# Prepear logging
# summary_writer = stable_nalu.writer.DummySummaryWriter()
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    f'{"-nac-" if args.nac_mul != "none" else ""}'
    f'{"n" if args.nac_mul == "normal" else ""}'
    f'{"s" if args.nac_mul == "safe" else ""}'
    f'{"s" if args.nac_mul == "max-safe" else ""}'
    f'{"t" if args.nac_mul == "trig" else ""}'
    f'{"m" if args.nac_mul == "mnac" else ""}'
    f'{"-nalu-" if (args.nalu_bias or args.nalu_two_nac or args.nalu_two_gate or args.nalu_mul != "normal" or args.nalu_gate != "normal") else ""}'
    f'{"b" if args.nalu_bias else ""}'
    f'{"2n" if args.nalu_two_nac else ""}'
    f'{"2g" if args.nalu_two_gate else ""}'
    f'{"s" if args.nalu_mul == "safe" else ""}'
    f'{"s" if args.nalu_mul == "max-safe" else ""}'
    f'{"t" if args.nalu_mul == "trig" else ""}'
    f'{"m" if args.nalu_mul == "mnac" else ""}'
    f'{"r" if args.nalu_gate == "regualized" else ""}'
    f'{"u" if args.nalu_gate == "gumbel" else ""}'
    f'{"uu" if args.nalu_gate == "obs-gumbel" else ""}'
    f'_d-{"".join(map(str, args.mnist_digits))}'
    f'_h-{args.mnist_outputs}'
    f'_op-{args.operation.lower()}'
    f'_oob-{"c" if args.nac_oob == "clip" else "r"}'
    f'_rs-{args.regualizer_scaling}-{args.regualizer_shape}'
    f'_eps-{args.mnac_epsilon}'
    f'_rl-{args.regualizer_scaling_start}-{args.regualizer_scaling_end}'
    f'_r-{args.regualizer}-{args.regualizer_z}-{args.regualizer_oob}'
    f'_m-{"s" if args.softmax_transform else "l"}-{args.model_simplification[0]}'
    f'_i-{args.interpolation_length}'
    f'_e-{"-".join(map(str, args.extrapolation_lengths))}'
    f'_b{args.batch_size}'
    f'_s{args.seed}'
    f'_nM{"T" if args.nmu_noise else f"F"}'
    f'_bA{"T" if args.beta_nau else f"F"}'
    f'_{args.train_criterion}{"-"+str(args.pcc2mse_iteration) if args.train_criterion == "pcc-mse" else ""}',
    remove_existing_data=args.remove_existing_data
)

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    torch.set_num_threads(int(os.environ['LSB_DJOB_NUMPROC']))

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
eps = torch.finfo().eps

# Setup datasets
dataset = stable_nalu.dataset.SequentialMnistDataset(
    operation=args.operation,
    use_cuda=args.cuda,
    seed=args.seed,
    mnist_digits=args.mnist_digits,
    num_workers=args.dataset_workers
)
dataset_train = dataset.fork(seq_length=args.interpolation_length, subset='train').dataloader(shuffle=True, batch_size=args.batch_size)
# Seeds are from random.org
dataset_train_full = dataset.fork(seq_length=args.interpolation_length, subset='train',
                                  seed=62379872).dataloader(shuffle=False)
dataset_valid = dataset.fork(seq_length=args.interpolation_length, subset='valid',
                                  seed=47430696).dataloader(shuffle=False)
dataset_test_extrapolations = [
    ( seq_length,
      dataset.fork(seq_length=seq_length, subset='test',
                   seed=88253339).dataloader(shuffle=False)
    ) for seq_length in args.extrapolation_lengths
]

# setup model
model = stable_nalu.network.SequentialMnistNetwork(
    args.layer_type,
    output_size=dataset.get_item_shape().target[-1],
    writer=summary_writer.every(100).verbose(args.verbose),
    mnist_digits=args.mnist_digits,
    mnist_outputs=args.mnist_outputs,
    model_simplification=args.model_simplification,
    softmax_transform=args.softmax_transform,
    nac_mul=args.nac_mul,
    nac_oob=args.nac_oob,
    regualizer_shape=args.regualizer_shape,
    regualizer_z=args.regualizer_z,
    mnac_epsilon=args.mnac_epsilon,
    nalu_bias=args.nalu_bias,
    nalu_two_nac=args.nalu_two_nac,
    nalu_two_gate=args.nalu_two_gate,
    nalu_mul=args.nalu_mul,
    nalu_gate=args.nalu_gate,
    nmu_noise=args.nmu_noise,
    nau_noise=args.nau_noise,
    beta_nau=args.beta_nau,
    noise_range=args.noise_range,
)
model.reset_parameters()
if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

seq_index = slice(None) if dataset.get_item_shape().target[0] is None else -1

def accuracy(y, t):
    return torch.mean((torch.round(y) == t).float())

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        mse_loss = 0
        acc_all = 0
        acc_last = 0
        for x, t in dataloader:
            # forward
            _, y = model(x)
            mse_loss += criterion(y[:,seq_index,:], t[:,seq_index,:]).item() * len(t)
            acc_all += accuracy(y[:,seq_index,:], t[:,seq_index,:]).item() * len(t)
            acc_last += accuracy(y[:,-1,:], t[:,-1,:]).item() * len(t)

        return (
            mse_loss / len(dataloader.dataset),
            acc_all / len(dataloader.dataset),
            acc_last / len(dataloader.dataset)
        )

print(model)
print('')
print(summary_writer.name)
print('')
utils.print_mnist_cell_weights(model)
print()

'''Resuming previous training'''
resume_epoch = 0
if args.load_checkpoint:
    resume_epoch = stable_nalu.writer.load_model(summary_writer.name, model, optimizer)
    if resume_epoch > args.max_epochs:
        raise ValueError(
            f'{args.max_iterations} must be larger than or equal to the loaded models resume epoch {resume_epoch}')
    print("Checkpoint loaded")
'''------------------'''
# Train model
global_step = 0
for epoch_i in range(resume_epoch, args.max_epochs + 1):
    for i_train, (x_train, t_train) in enumerate(dataset_train):
        global_step += 1
        summary_writer.set_iteration(global_step)
        summary_writer.add_scalar('epoch', epoch_i)

        # Prepear model
        model.set_parameter('tau', max(0.5, math.exp(-1e-5 * global_step)))
        optimizer.zero_grad()

        # Log validation
        if epoch_i % 5 == 0 and i_train == 0:
            (train_full_mse,
             train_full_acc_all,
             train_full_acc_last) = test_model(dataset_train_full)
            summary_writer.add_scalar('metric/train/mse', train_full_mse)
            summary_writer.add_scalar('metric/train/acc/all', train_full_acc_all)
            summary_writer.add_scalar('metric/train/acc/last', train_full_acc_last)

            (valid_mse,
             valid_acc_all,
             valid_acc_last) = test_model(dataset_valid)
            summary_writer.add_scalar('metric/valid/mse', valid_mse)
            summary_writer.add_scalar('metric/valid/acc/all', valid_acc_all)
            summary_writer.add_scalar('metric/valid/acc/last', valid_acc_last)

            for seq_length, dataloader in dataset_test_extrapolations:
                (test_extrapolation_mse,
                 test_extrapolation_acc_all,
                 test_extrapolation_acc_last) = test_model(dataloader)
                summary_writer.add_scalar(f'metric/test/extrapolation/{seq_length}/mse', test_extrapolation_mse)
                summary_writer.add_scalar(f'metric/test/extrapolation/{seq_length}/acc/all', test_extrapolation_acc_all)
                summary_writer.add_scalar(f'metric/test/extrapolation/{seq_length}/acc/last', test_extrapolation_acc_last)

        # forward
        with summary_writer.force_logging(epoch_i % 5 == 0 and i_train == 0):
            mnist_y_train, y_train = model(x_train)
        regualizers = model.regualizer()

        if (args.regualizer_scaling == 'linear'):
            r_w_scale = max(0, min(1, (
                (global_step - args.regualizer_scaling_start) /
                (args.regualizer_scaling_end - args.regualizer_scaling_start)
            )))
        elif (args.regualizer_scaling == 'exp'):
            r_w_scale = 1 - math.exp(-1e-5 * global_step)

        #'mse loss'
        mse_loss = criterion(y_train[:, seq_index, :], t_train[:, seq_index, :])

        #'pcc loss'
        # Acknowledgement: https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739
        vx = y_train[:, seq_index, :] - torch.mean(y_train[:, seq_index, :])
        vy = t_train[:, seq_index, :] - torch.mean(t_train[:, seq_index, :])
        r = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(
            torch.sum(vy ** 2))) + eps)  # + eps to avoid denominator = 0
        pcc_loss = 1 - r

        #'mean abs percentage error (mape) loss'
        mape_loss = torch.mean(torch.abs((t_train[:, seq_index, :] - y_train[:, seq_index, :]) / t_train[:, seq_index, :]))

        loss_train_regualizer = args.regualizer * r_w_scale * regualizers['W'] + \
                                regualizers['g'] + \
                                args.regualizer_z * regualizers['z'] + \
                                args.regualizer_oob * regualizers['W-OOB']

        loss_train_criterion = stable_nalu.functional.get_train_criterion(args.train_criterion, mse_loss, pcc_loss,
                                                                          mape_loss, global_step,
                                                                          args.pcc2mse_iteration)
        loss_train = loss_train_criterion + loss_train_regualizer

        # Log loss
        summary_writer.add_scalar('loss/train/accuracy/all', accuracy(y_train[:,seq_index,:], t_train[:,seq_index,:]))  # if seqlen= (None, None, None) then will give everything
        summary_writer.add_scalar('loss/train/accuracy/last', accuracy(y_train[:,-1,:], t_train[:,-1,:]))
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
        if epoch_i % 5 == 0 and i_train == 0:
            summary_writer.add_tensor('MNIST/train',
                                      torch.cat([mnist_y_train[:,0,:], t_train[:,0,:]], dim=1))
            print('train %d: %.5f, full: %.5f, %.3f (acc[last]), valid: %.5f, %.3f (acc[last])' % (
                epoch_i, loss_train_criterion, train_full_mse, train_full_acc_last, valid_mse, valid_acc_last
            ))

        # Optimize model
        if loss_train.requires_grad:
            loss_train.backward()
            optimizer.step()
        model.optimize(loss_train_criterion)

        # Log gradients if in verbose mode
        with summary_writer.force_logging(epoch_i % 5 == 0 and i_train == 0):
            model.log_gradients()

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - validation: {valid_mse}')

utils.print_mnist_cell_weights(model)

model.writer._root.close()  # fix - close summary writer before saving model to avoid thread locking issues

# Use saved weights to visualize the intermediate values.
stable_nalu.writer.save_model_checkpoint(summary_writer.name, epoch_i + 1, model, optimizer,
                                         {'torch': torch.get_rng_state(),
                                          'numpy': np.random.get_state()}
                                         )