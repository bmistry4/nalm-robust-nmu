import math
import os
from decimal import Decimal

import numpy as np
import torch

import misc.utils as utils
import stable_nalu
import stable_nalu.functional.regualizer as Regualizer
from experiments.single_layer_args import print_parser_configuration, create_base_parser, robustness_experiments_args, \
    print_robustness_params, parse_sample_distribution_args

# Parse arguments
parser = create_base_parser()
robustness_experiments_args(parser)
args = parser.parse_args()
args.sample_distribution = parse_sample_distribution_args(args.sample_distribution)

utils.set_pytorch_precision(args.pytorch_precision)
setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)

# Print configuration
print_parser_configuration(args)
print_robustness_params(args)


def get_sample_distribution_writer_value():
    family = args.sample_distribution[0]
    if family == 'uniform':
        return 'u'
    elif family == 'truncated-normal':
        mean = args.sample_distribution[1]
        std = args.sample_distribution[2]
        return f'tn-{mean}-{std}'
    elif family == 'exponential':
        scale = args.sample_distribution[1]
        return f'e-{str(scale)}'
    elif family == 'benford':
        return 'b'


def get_npu_Wr_init_writer_value():
    if args.npu_Wr_init == 'xavier-uniform':
        return 'xu'
    elif args.npu_Wr_init == 'xavier-uniform-constrained':
        return 'xuc'
    else:
        raise ValueError(f'Invalid arg ({args.npu_Wr_init}) given for npu_Wr_init')


def get_train_criterion_writer_value():
    val = args.train_criterion
    if args.train_criterion == 'pcc-mse':
        val += f'-{str(args.pcc2mse_iteration)}'
    return val


# Prepare logging
# summary_writer = stable_nalu.writer.DummySummaryWriter()
# TODO: missing args: output_size
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    # f'{"-nac-" if args.nac_mul != "none" else ""}'
    # f'{"n" if args.nac_mul == "normal" else ""}'
    # f'{"s" if args.nac_mul == "safe" else ""}'
    # f'{"s" if args.nac_mul == "max-safe" else ""}'
    # f'{"t" if args.nac_mul == "trig" else ""}'
    # f'{"m" if args.nac_mul == "mnac" else ""}'
    # f'{"npu" if args.nac_mul == "npu" else ""}'
    # f'{"npur" if args.nac_mul == "real-npu" else ""}'
    # f'{"-nalu-" if (args.nalu_bias or args.nalu_two_nac or args.nalu_two_gate or args.nalu_mul != "normal" or args.nalu_gate != "normal") else ""}'
    f'{"-b" if args.nalu_bias and args.layer_type == "NALU" else ""}'
    f'{"-2n" if args.nalu_two_nac and args.layer_type == "NALU" else ""}'
    f'{"-2g" if args.nalu_two_gate and args.layer_type == "NALU" else ""}'
    f'{"-s" if args.nalu_mul == "safe" and args.layer_type == "NALU" else ""}'
    f'{"-s" if args.nalu_mul == "max-safe" and args.layer_type == "NALU" else ""}'
    f'{"-t" if args.nalu_mul == "trig" and args.layer_type == "NALU" else ""}'
    f'{"-m" if args.nalu_mul == "mnac" and args.layer_type == "NALU" else ""}'
    f'{"-r" if args.nalu_gate == "regualized" and args.layer_type == "NALU" else ""}'
    f'{"-u" if args.nalu_gate == "gumbel" and args.layer_type == "NALU" else ""}'
    f'{"-uu" if args.nalu_gate == "obs-gumbel" and args.layer_type == "NALU" else ""}'
    f'{"-sS" if args.nru_div_mode == "div-sepSign" and args.layer_type == "NRU" else ""}'
    f'_op-{args.operation.lower()}'
    f'_oob-{"c" if args.oob_mode == "clip" else "r"}'
    f'_rs-{args.regualizer_scaling}-{args.regualizer_shape}'
    f'_eps-{args.mnac_epsilon}'
    f'_rl-{args.regualizer_scaling_start}-{args.regualizer_scaling_end}'
    f'_r-{args.regualizer}-{args.regualizer_z}-{args.regualizer_oob}'
    f'_i-{args.interpolation_range[0]}-{args.interpolation_range[1]}'
    f'_e-{args.extrapolation_range[0]}-{args.extrapolation_range[1]}'
    f'_z-{"simple" if args.simple else f"{args.input_size}-{args.subset_ratio}-{args.overlap_ratio}"}'
    f'_b{args.batch_size}'
    f'_s{args.seed}'
    f'_h{args.hidden_size}'
    f'_z{args.num_subsets}'
    f'_lr-{args.optimizer}-{"%.5f" % args.learning_rate}-{args.momentum}'
    f'_D-{get_sample_distribution_writer_value()}'
    f'_n-{args.noise_range[0]}-{args.noise_range[1]}'
    f'_L1{"T" if args.regualizer_l1 else f"F"}'
    f'_rb-{args.regualizer_beta_start}-{args.regualizer_beta_end}-{args.regualizer_beta_step}-{args.regualizer_beta_growth}'
    f'_rWnpu-{args.regualizer_npu_w}-{args.realnpu_reg_type[0]}'
    f'_rg-{args.regualizer_gate}'
    f'_r{"H" if args.reg_scale_type == "heim" else f"M"}'
    f'_clip-{args.npu_clip if args.npu_clip != "none" else args.npu_clip[0]}'
    f'_WrI-{get_npu_Wr_init_writer_value()}'
    # f'_p-{args.pytorch_precision}'
    f'_nM{"T" if args.nmu_noise else f"F"}'
    # f'_nA{"T" if args.nau_noise else f"F"}'
    # f'_Bnau{"T" if args.beta_nau else f"F"}'
    f'_gn-{args.clip_grad_norm if args.clip_grad_norm != None else f"F"}'
    f'_TB-{args.log_interval}'
    f'_{get_train_criterion_writer_value()}',
    remove_existing_data=args.remove_existing_data
)

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    torch.set_num_threads(int(os.environ['LSB_DJOB_NUMPROC']))

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# set epsilon for numerical stability
eps = torch.finfo().eps

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
    dist_params=args.sample_distribution,
)
print(f'  -')
print(f'  - dataset: {dataset.print_operation()}')
dataset_train = iter(dataset.fork(sample_range=args.interpolation_range).dataloader(batch_size=args.batch_size))

# setup model
model = stable_nalu.network.SingleLayerNetwork(
    args.layer_type,
    input_size=dataset.get_input_size(),
    output_size=args.output_size,
    writer=summary_writer.every(args.log_interval).verbose(args.verbose),
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
    regualizer_gate=args.regualizer_gate,
    regualizer_npu_w=args.regualizer_npu_w,
    nmu_noise=args.nmu_noise,
    noise_range=args.noise_range,
    nau_noise=args.nau_noise,
    beta_nau=args.beta_nau,
    npu_clip=args.npu_clip,
    npu_Wr_init=args.npu_Wr_init,
    trash_cell=False,  # TODO: remove?
    nru_div_mode=args.nru_div_mode,
    realnpu_reg_type=args.realnpu_reg_type,
    use_robustness_exp_logging=True
)

model.reset_parameters()
if args.cuda:
    model.cuda()

criterion = torch.nn.MSELoss()
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    raise ValueError(f'{args.optimizer} is not a valid optimizer algorithm')


def test_model(data):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        model.eval()
        x, t = data
        err = criterion(model(x), t)
        model.train()
        return err


# Train model
print(model)
print('')
print(summary_writer.name)
print('')
# only print inits of small models
utils.print_model_params(model) if args.input_size <= 10 else None
print()

use_npu_scaling = args.regualizer_l1 or (args.regualizer_npu_w and args.reg_scale_type == 'heim') \
                  or (args.regualizer_gate and args.reg_scale_type == 'heim')
if use_npu_scaling:
    # Decimal type required to avoid accumulation of fp precision errors when multiplying by growth factor
    args.regualizer_beta_start = Decimal(str(args.regualizer_beta_start))
    # Decimal and fp arithmetic don't mix so beta end must also be a decimal
    args.regualizer_beta_end = Decimal(str(args.regualizer_beta_end))
r_l1_scale = args.regualizer_beta_start

'''Resuming previous training'''
resume_epoch = 0
if args.load_checkpoint:
    resume_epoch = stable_nalu.writer.load_model(summary_writer.name, model, optimizer)
    if resume_epoch > args.max_iterations:
        raise ValueError(
            f'{args.max_iterations} must be larger than or equal to the loaded models resume epoch {resume_epoch}')
    if resume_epoch != 0:  # FIXME this line is redundant?
        for i, j in zip(range(resume_epoch), dataset_train):
            (x_train, t_train) = j
    print("Checkpoint loaded")
    print('train %d: %.5f' % (resume_epoch, test_model((x_train, t_train))))
'''------------------'''
for epoch_i, (x_train, t_train) in zip(range(resume_epoch, args.max_iterations + 1), dataset_train):
    summary_writer.set_iteration(epoch_i)
    # model.log_learnable_parameters()      # log each learnable parameter element as a SCALAR on tensorboard

    # Prepear model
    model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()

    # forward
    y_train = model(x_train)
    regualizers = model.regualizer()  # logs 3 reg metrics to tensorboard if verbose

    if (args.regualizer_scaling == 'linear'):
        r_w_scale = max(0, min(1, (
                (epoch_i - args.regualizer_scaling_start) /
                (args.regualizer_scaling_end - args.regualizer_scaling_start)
        )))
    elif (args.regualizer_scaling == 'exp'):
        r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

    l1_loss = 0
    if args.regualizer_l1:
        l1_loss = Regualizer.l1(model.parameters())
        if args.verbose:
            summary_writer.add_scalar('L1/train/L1-loss', l1_loss)

    if use_npu_scaling:
        # the beta_start value will be updated accordingly to be the correct beta value for the epoch.
        # It is done this way to avoid having initalise another variable outside the epoch loop
        if args.regualizer_beta_start <= args.regualizer_beta_end:
            if epoch_i % args.regualizer_beta_step == 0 and epoch_i != 0:
                if args.regualizer_beta_start < args.regualizer_beta_end:
                    args.regualizer_beta_start *= args.regualizer_beta_growth
        else:
            if epoch_i % args.regualizer_beta_step == 0 and epoch_i != 0:
                if args.regualizer_beta_start > args.regualizer_beta_end:
                    args.regualizer_beta_start /= args.regualizer_beta_growth

        r_l1_scale = float(args.regualizer_beta_start)  # Decimal doesn't work for tensorboard or mixed fp arithmetic
        if args.verbose:
            summary_writer.add_scalar('L1/train/beta', r_l1_scale)

    # mse loss
    mse_loss = criterion(y_train, t_train)
    loss_train_criterion = mse_loss

    # on assumption most runs will use mse so no need to calc the other stats
    if args.train_criterion != 'mse':
        # pcc loss
        # Acknowledgement: https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739
        vx = y_train - torch.mean(y_train)
        vy = t_train - torch.mean(t_train)

        """
        Acknowledgement: https://www.johndcook.com/blog/2008/11/05/how-to-calculate-pearson-correlation-accurately/
        Use population stats (i.e., no Bassels' correction) and +eps to avoid div by 0.
        """
        n = t_train.shape[0]
        sx = torch.sqrt((torch.sum(vx ** 2) / n).clamp(eps))
        sy = torch.sqrt((torch.sum(vy ** 2) / n).clamp(eps))
        r = torch.sum((vx / (sx + eps)) * (vy / (sy + eps))) / n
        # assert -1 <= r <= 1, f"r (={r}) value is not in valid range of [-1,1]"
        pcc_loss = 1 - r

        # mean abs percentage error
        mape_loss = torch.mean(torch.abs((t_train - y_train) / t_train))

        loss_train_criterion = stable_nalu.functional.get_train_criterion(args.train_criterion, mse_loss, pcc_loss, mape_loss,
                                                                          epoch_i, args.pcc2mse_iteration)

    loss_train_regualizer = args.regualizer * r_w_scale * regualizers['W'] + \
                            regualizers['g'] + \
                            args.regualizer_z * regualizers['z'] + \
                            args.regualizer_oob * regualizers['W-OOB'] + \
                            args.regualizer_l1 * r_l1_scale * l1_loss + \
                            args.regualizer_npu_w * (r_l1_scale if args.reg_scale_type == 'heim' else r_w_scale) * regualizers['W-NPU'] + \
                            args.regualizer_gate * (r_l1_scale if args.reg_scale_type == 'heim' else r_w_scale) * regualizers['g-NPU']

    loss_train = loss_train_criterion + loss_train_regualizer

    # Log loss
    if args.verbose or epoch_i % args.log_interval == 0:
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
    # only log to console every 1000 epochs
    if epoch_i % 1000 == 0:
        print('train %d: %.5f, reg loss: %.5f, total: %.5f' % (epoch_i, loss_train_criterion, loss_train_regualizer, loss_train))


    # Optimize model
    if loss_train.requires_grad:
        loss_train.backward()
        if args.clip_grad_norm != None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
    model.optimize(loss_train_criterion)

    # Log gradients if in verbose mode
    if args.verbose and epoch_i % args.log_interval == 0:
        model.log_gradients()
        # model.log_gradient_elems()

# Compute validation loss
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print()
utils.print_model_params(model)

if not args.no_save:
    model.writer._root.close()  # fix - close summary writer before saving model to avoid thread locking issues
    # Use saved weights to visualize the intermediate values.
    stable_nalu.writer.save_model_checkpoint(summary_writer.name, epoch_i + 1, model, optimizer,
                                             {'torch': torch.get_rng_state(),
                                              'numpy': np.random.get_state()}
                                             )
