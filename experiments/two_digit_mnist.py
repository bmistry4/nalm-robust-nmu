"""
Ack:
The task

"""
import os
os.environ['MPLCONFIGDIR'] = '/tmp'
import argparse
import ast
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import stable_nalu.writer as writer
from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.layer import ReRegualizedLinearMNACLayer
from stable_nalu.layer import ReRegualizedLinearNACLayer
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler

parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--id',
                    action='store',
                    default=-1,
                    type=int,
                    help='Unique id to identify experiment')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')
parser.add_argument('--data-path',
                    action='store',
                    default='../stable_nalu/dataset/data/two_digit_mnist',
                    type=str,
                    help='Where the (mnist) data should be stored')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=['add', 'mul'],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul')

parser.add_argument('--max-epochs',
                    action='store',
                    default=100,
                    type=int,
                    help='Specify the max number of epochs to use')
parser.add_argument('--samples-per-permutation',
                    action='store',
                    default=1000,
                    type=int,
                    help='Number of samples per permutation (e.g. there are 90 permutations in the train set so 1000 * 90).')
parser.add_argument('--num-folds',
                    action='store',
                    default=10,
                    type=int,
                    help='Number of folds for cross-val')
parser.add_argument('--batch-size',
                    action='store',
                    default=128,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--learning-rate',
                    action='store',
                    default=1e-3,
                    type=float,
                    help='Specify the learning-rate')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='two_digit_mnist',
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

parser.add_argument('--log-interval',
                    action='store',
                    default=1,
                    type=int,
                    help='Log to tensorboard every X epochs.')
parser.add_argument('--mb-log-interval',
                    action='store',
                    default=100,
                    type=int,
                    help='Log to tensorboard every X minibatches.')
parser.add_argument('--dataset-workers',
                    action='store',
                    default=0,
                    type=int,
                    help='Number of workers for multi-process data loading')

parser.add_argument('--use-nalm',
                    action='store_true',
                    default=False,
                    help=f'')
parser.add_argument('--learn-labels2out',
                    action='store_true',
                    default=False,
                    help=f'If the last layer should have learnable params (True) or just apply the correct operation (False)')

parser.add_argument('--regualizer-scaling-start',
                    action='store',
                    default=50,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=75,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer',
                    action='store',
                    default=10,
                    type=float,
                    help='Specify the regualization lambda to be used')

parser.add_argument('--beta-nau',
                    action='store_true',
                    default=False,
                    help='Have nau weights initialised using a beta distribution B(7,7)')
parser.add_argument('--nau-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies additive noise from a ~U[1,5] during training.')
parser.add_argument('--nmu-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies multiplicative noise from a ~U[1,5] during training. Aids with failure ranges on a vinilla NMU.')
parser.add_argument('--noise-range',
                    action='store',
                    default=[1, 5],
                    type=ast.literal_eval,
                    help='Range at which the noise for applying stochasticity is taken from. (Originally for sNMU.)')

parser.add_argument('--no-save',
                    action='store_true',
                    default=False,
                    help='Do not save model at the end of training')
parser.add_argument('--load-checkpoint',
                    action='store_true',
                    default=False,
                    help='Loads a saved checkpoint and resumes training')

parser.add_argument('--rgb',
                    action='store_true',
                    default=False,
                    help='If images are in colour (rgb). Used so we know the number of colour channels.')


def create_tb_writer(args, fold_idx, use_dummy=False):
    if use_dummy:
        return writer.DummySummaryWriter()
    else:
        return writer.SummaryWriter(
            name=
            f'{args.name_prefix}/{args.id}'
            f'_f{fold_idx}'
            f'_op-{args.operation}'
            f'_nalm{str(args.use_nalm)[0]}'
            f'_learnL{str(args.learn_labels2out)[0]}'
            f'_s{args.seed}',
            remove_existing_data=args.remove_existing_data
        )


class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''

    def __call__(self, image):
        return image


class Img2LabelsRegression(ExtendedTorchModule):
    """
    follows the architecture from https://github.com/mdbloice/MNIST-Calculator/blob/main/MNIST-Calculator.ipynb which
    is a modified version for https://github.com/pytorch/examples/blob/master/mnist/main.py (which implements a
    convolutional NN similar to the original LeNet5).
    """
    # TODO - this one is good for addition but struggles on mul (based off seq mnist convnet)
    # def __init__(self, **kwargs):
    #     super(Img2LabelsRegression, self).__init__('cnn_reg', **kwargs)
    #     self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
    #     self.bn1 = torch.nn.BatchNorm2d(20)
    #     self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
    #     self.bn2 = torch.nn.BatchNorm2d(50)
    #     self.dropout1 = nn.Dropout(0.25)
    #     self.dropout2 = nn.Dropout(0.5)
    #     self.fc1 = torch.nn.Linear(50*4*11, 500)  # Default was 128
    #     self.fc2 = nn.Linear(500, 2)
    #     # self.fc3 = nn.Linear(100, 2)  # MNIST-calculator example only has 1 output to predict the final result, however we want 2 outputs (for the img lables) as this is an intermediary module.
    #
    # def forward(self, x):
    #     # x shape: [B, C=1, H=28, W=56]
    #     x = self.bn1(torch.nn.functional.relu(self.conv1(x)))
    #     x = torch.nn.functional.max_pool2d(x, 2, 2)
    #     x = self.bn2(torch.nn.functional.relu(self.conv2(x)))
    #     x = torch.nn.functional.max_pool2d(x, 2, 2)
    #     x = torch.flatten(x, 1)
    #     x = torch.nn.functional.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     output = x
    #     # output = F.log_softmax(x, dim=1)
    #     # output = x.sum(1)
    #     return output

    # TODO - based off mnist example - arch used for first conv- runs (id 1-8); assume 1 colour channel
    def __init__(self, **kwargs):
        super(Img2LabelsRegression, self).__init__('cnn_reg', **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(19968, 256)  # Default was 128
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 2)  # MNIST-calculator example only has 1 output to predict the final result, however we want 2 outputs (for the img lables) as this is an intermediary module.

    def forward(self, x):
        # x shape: [B, C=1, H=28, W=56]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        # output = x.sum(1)
        return output   # [B,2]


# FIXME - keeps predicting the same output labels no matter the input value
class Img2LabelsClassification(ExtendedTorchModule):
    """
    follows the architecture from https://github.com/mdbloice/MNIST-Calculator/blob/main/MNIST-Calculator.ipynb which
    is a modified version for https://github.com/pytorch/examples/blob/master/mnist/main.py (which implements a
    convolutional NN similar to the original LeNet5).
    """

    # TODO - this one is good for addition but struggles on mul (based off seq mnist convnet)
    def __init__(self, **kwargs):
        super(Img2LabelsClassification, self).__init__('cnn_clf', **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.bn2 = torch.nn.BatchNorm2d(50)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(50*4*11, 500)  # Default was 128
        self.fc2 = nn.Linear(500, 20)   # 2 sets of 10 where 10 reps labels 0-9

    def forward(self, x):
        # x shape: [B, C=1, H=28, W=56]
        x = self.bn1(torch.nn.functional.relu(self.conv1(x)))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = self.bn2(torch.nn.functional.relu(self.conv2(x)))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        # split into 2 clf tasks (1 clf for the left img and 1 clf for the right img)
        x = x.reshape(-1, 2, 10)  # [B,2,10]

        output = F.softmax(x, dim=2)  # TODO scale by temp?
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor)  # [10] numbers to index from
        # softargmax
        output = output @ digits  # [B,2,10] [10] = [B,2,1]
        output = output.squeeze()
        return output # [B,2]


    # # TODO - based off mnist example - arch used for first conv- runs (id 1-8); assume 1 colour channel
    # def __init__(self, **kwargs):
    #     super(Img2LabelsClassification, self).__init__('cnn_clf', **kwargs)
    #     self.conv1 = nn.Conv2d(1, 32, 3, 1)
    #     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    #     self.dropout1 = nn.Dropout(0.25)
    #     self.dropout2 = nn.Dropout(0.5)
    #     self.fc1 = nn.Linear(19968, 256)  # Default was 128
    #
    #     self.fc2 = nn.Linear(256, 20)   # 2 sets of 10 where 10 reps labels 0-9
    #
    # def forward(self, x):
    #     # x shape: [B, C=1, H=28, W=56]
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = F.max_pool2d(x, 2)
    #     x = self.dropout1(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.dropout2(x)
    #     x = self.fc2(x)
    #     x = F.relu(x)
    #
    #     # split into 2 clf tasks (1 clf for the left img and 1 clf for the right img)
    #     x = x.reshape(-1, 2, 10)    # [B,2,10]
    #     x = F.softmax(x, dim=2)     # TODO scale by temp?
    #     digits = torch.arange(0, 10, 1).type(torch.FloatTensor)  # [10] numbers to index from
    #     # softargmax
    #     x = x @ digits      # [B,2,10] [10] = [B,2,1]
    #     output = x.squeeze()
    #     return output   # [B,2]

class Img2LabelsIndepImgClf(ExtendedTorchModule):
    """
    Encoder for mnist single mnist digits.
    Assumes input given is a img with 2 mnnist digits
    Output will return 2 labels (1 f.e. digit)

    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsIndepImgClf, self).__init__('cnn_1_digit_clf', **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.device = device

    def forward(self, x):
        # change 1 img containing 2 digits into 2 imgs with 1 digit
        x = torch.cat(x.split(x.shape[-1] // 2, -1), 0)  # [2*B, C=1, 28, 28]
        x = self.conv1(x)   # [2*B, 32, 26, 26]
        x = F.relu(x)
        x = self.conv2(x)   # [2*B, 64, 24, 24]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # [2*B, 64, 12, 12]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)     # [2*B, 9216]
        x = self.fc1(x)             # [2*B, 128]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)             # [2*B, 10]
        # [2B,10] -> [2B,1] -> [B,2]
        output = F.softmax(x, dim=1)  # [2*B,10] # TODO scale by temp?
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)   # [10] numbers to index from
        # softargmax
        output = output @ digits  # [2*B,10] [10] = [2*B]
        output = output.unsqueeze(1)
        # recombine the original digits back to the 2-digit img
        output = torch.cat(output.split(output.shape[0] // 2, 0), -1)  # [B, 2]
        return output

# class Img2LabelsSpatialTransformer(ExtendedTorchModule):
#     """
#     Use spatial transformer network for learning localisation and selection of each digit and then additional
#     classification network to get the two labels.
#     Based off Pytorch tutorial: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
#     Convert a image into 2 channels (1 per digit) and pass it through 2 different spatial transformers to get 2
#     transformations which get applied indep and concatenated resulting in a 4 channel f.map which gets pushed though
#     additional modules until it becomes of shape [B,2,10] to be used as a classifier f.e. digit.
#     """
#     def __init__(self, device, **kwargs):
#         super(Img2LabelsSpatialTransformer, self).__init__('stn_clf', **kwargs)
#
#         self.conv1 = nn.Conv2d(4, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(880, 50)
#         self.fc2 = nn.Linear(50, 20)
#
#         # Spatial transformer localization-network
#         self.loc1 = nn.Sequential(
#             nn.Conv2d(2, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc1 = nn.Sequential(
#             nn.Linear(300, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Spatial transformer localization-network
#         self.loc2 = nn.Sequential(
#             nn.Conv2d(2, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc2 = nn.Sequential(
#             nn.Linear(300, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc1[2].weight.data.zero_()
#         self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#         self.fc_loc2[2].weight.data.zero_()
#         self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#         self.device = device
#
#     # Spatial transformer network forward function
#     def stn(self, x):
#         # x = [B,2, 56, 28]
#         xs1 = self.loc1(x)
#         # [B, 300]
#         xs1 = xs1.view(-1, 10*3*10)
#         # [B, 6]
#         theta1 = self.fc_loc1(xs1)
#         # [B, 2, 3]
#         theta1 = theta1.view(-1, 2, 3)
#         # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
#         grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
#         # [B, C, H, W] = grid_sample([B, 2, H, W], [B,H,W,2])
#         x1 = F.grid_sample(x, grid1, align_corners=False)
#
#         xs2 = self.loc2(x)
#         xs2 = xs2.view(-1, 10*3*10)
#         theta2 = self.fc_loc2(xs2)
#         theta2 = theta2.view(-1, 2, 3)
#         grid = F.affine_grid(theta2, x.size(), align_corners=False)
#         x2 = F.grid_sample(x, grid, align_corners=False)
#
#         return x1, x2
#
#     def forward(self, x):
#         bsz = x.shape[0]
#         # make img into 2 channel -> [B,2,H,W]
#         x = x.repeat(1, 2, 1, 1)
#         # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,56]
#         x_st1, x_st2 = self.stn(x)
#         x_st = torch.cat([x_st1, x_st2], dim=1)  # [B,4,28,56]
#
#         # Perform the usual forward pass
#         x_st = F.relu(F.max_pool2d(self.conv1(x_st), 2))                    # [B,10,12,26]
#         x_st = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x_st)), 2))   # [B,20, 4, 11]
#         x_st = x_st.view(bsz, 880)                                          # [B, 880]
#         x_st = F.relu(self.fc1(x_st))                                       # [B, 880] -> [B,50]
#         x_st = F.dropout(x_st, training=self.training)
#
#         x_st = self.fc2(x_st)                                               # [B, 50] -> [B, 20]
#         x_st = x_st.reshape(-1, 2, 10)                                      # [B,20] -> [B, 2, 10]
#         output = F.softmax(x_st, dim=-1)                                    # [B, 2, 10]
#         digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from
#         # softargmax
#         output = output @ digits                                            # [B, 2, 10] [10] = [B,2]
#         return output

# class Img2LabelsSpatialTransformer(ExtendedTorchModule):
#     """
#     Use spatial transformer network for learning localisation and selection of each digit and then additional
#     classification network to get the two labels.
#     Based off Pytorch tutorial: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
#     Creates a 2 channel img (1 channel per digit) and passes it through 2 different STs independantly. The two resulting
#     feature maps are individually passed through a set of transformations to get logits which are used for
#     classification.
#     """
#     def __init__(self, device, **kwargs):
#         super(Img2LabelsSpatialTransformer, self).__init__('stn_clf', **kwargs)
#
#         self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(880, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#         # Spatial transformer localization-network
#         self.loc1 = nn.Sequential(
#             nn.Conv2d(2, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc1 = nn.Sequential(
#             nn.Linear(300, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Spatial transformer localization-network
#         self.loc2 = nn.Sequential(
#             nn.Conv2d(2, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc2 = nn.Sequential(
#             nn.Linear(300, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc1[2].weight.data.zero_()
#         self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#         self.fc_loc2[2].weight.data.zero_()
#         self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#         self.device = device
#
#     # Spatial transformer network forward function
#     def stn(self, x):
#         # x = [B,2, 56, 28]
#         xs1 = self.loc1(x)
#         # [B, 300]
#         xs1 = xs1.view(-1, 10*3*10)
#         # [B, 6]
#         theta1 = self.fc_loc1(xs1)
#         # [B, 2, 3]
#         theta1 = theta1.view(-1, 2, 3)
#         # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
#         grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
#         # [B, C, H, W] = grid_sample([B, 2, H, W], [B,H,W,2])
#         x1 = F.grid_sample(x, grid1, align_corners=False)
#
#         xs2 = self.loc2(x)
#         xs2 = xs2.view(-1, 10*3*10)
#         theta2 = self.fc_loc2(xs2)
#         theta2 = theta2.view(-1, 2, 3)
#         grid = F.affine_grid(theta2, x.size(), align_corners=False)
#         x2 = F.grid_sample(x, grid, align_corners=False)
#
#         return x1, x2
#
#     def forward(self, x):
#         bsz = x.shape[0]
#         # make img into 2 channel -> [B,2,H,W]
#         x = x.repeat(1, 2, 1, 1)
#         # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,56]
#         x_st1, x_st2 = self.stn(x)
#
#         # Perform the usual forward pass for 1st ST
#         x_st1 = F.relu(F.max_pool2d(self.conv1(x_st1), 2))                    # [B,10,12,26]
#         x_st1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x_st1)), 2))   # [B,20, 4, 11]
#         x_st1 = x_st1.view(bsz, 880)                                          # [B, 880]
#         x_st1 = F.relu(self.fc1(x_st1))                                       # [B, 880] -> [B,50]
#         x_st1 = F.dropout(x_st1, training=self.training)
#         x_st1 = self.fc2(x_st1)                                               # [B, 50] -> [B, 10]
#         x_st1 = x_st1.reshape(-1, 1, 10)                                      # [B,10] -> [B, 1, 10]
#         x_st1 = F.softmax(x_st1, dim=-1)                                    # [B, 1, 10]
#
#         # Perform the usual forward pass for 2nd ST
#         x_st2 = F.relu(F.max_pool2d(self.conv1(x_st2), 2))  # [B,10,12,26]
#         x_st2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x_st2)), 2))  # [B,20, 4, 11]
#         x_st2 = x_st2.view(bsz, 880)  # [B, 880]
#         x_st2 = F.relu(self.fc1(x_st2))  # [B, 880] -> [B,50]
#         x_st2 = F.dropout(x_st2, training=self.training)
#         x_st2 = self.fc2(x_st2)  # [B, 50] -> [B, 10]
#         x_st2 = x_st2.reshape(-1, 1, 10)  # [B,10] -> [B, 1, 10]
#         x_st2 = F.softmax(x_st2, dim=-1)  # [B, 1, 10]
#
#         output = torch.cat((x_st1, x_st2), dim=1)
#         digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from
#         # softargmax
#         output = output @ digits                                            # [B, 2, 10] [10] = [B,2]
#         return output


class Img2LabelsSpatialTransformer(ExtendedTorchModule):
    """
    Use spatial transformer network for learning localisation and selection of each digit and then additional
    classification network to get the two labels.
    Based off Pytorch tutorial: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    Convert a image into 2 channels (1 per digit) and pass it through 2 different spatial transformers to get 2
    transformations which get applied indep and concatenated resulting in a 4 channel f.map which gets pushed though
    additional modules until it becomes of shape [B,2,10] to be used as a classifier f.e. digit.
    Uses an attention based ST matrix so the transformation is less expressive (c.f. matrix for attn and affine) but
    only requires 3 params to learn (instead of 6).
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformer, self).__init__('stn_clf', **kwargs)

        self.conv1 = nn.Conv2d(4, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(880, 50)
        self.fc2 = nn.Linear(50, 20)

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(300, 32),
            nn.ReLU(True),
            nn.Linear(32, 3)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(300, 32),
            nn.ReLU(True),
            nn.Linear(32, 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))  # set scaling to start at 1 (128/128) -> (before ST/ after ST)
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))  # set scaling to start at 1 (128/128) -> (before ST/ after ST)
        self.device = device

    # Spatial transformer network forward function
    def stn(self, x):
        # x = [B,2, 56, 28]
        xs1 = self.loc1(x)
        # [B, 300]
        xs1 = xs1.view(-1, 10*3*10)
        # [B, 3]
        theta1 = self.fc_loc1(xs1)
        # [B,1]
        scale1 = theta1[:, 0].unsqueeze(1)   # get the scaling factor
        # [B,2]
        scale_mat1 = torch.cat((scale1, scale1), 1)    # will be creating a diagonal matrix from this
        # [B,2,1]
        translation1 = theta1[:, 1:].unsqueeze(2)
        # [B,2,3] = cat([B,2,2], [B,2,1])
        theta1 = torch.cat((torch.diag_embed(scale_mat1), translation1), 2)
        # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)    # TODO - downsampling?
        # [B, C, H, W] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(-1, 10 * 3 * 10)
        theta2 = self.fc_loc2(xs2)
        scale2 = theta2[:, 0].unsqueeze(1)
        scale_mat2 = torch.cat((scale2, scale2), 1)
        translation2 = theta2[:, 1:].unsqueeze(2)
        theta2 = torch.cat((torch.diag_embed(scale_mat2), translation2), 2)
        grid2 = F.affine_grid(theta2, x.size(), align_corners=False)    # TODO - downsampling?
        x2 = F.grid_sample(x, grid2, align_corners=False)

        return x1, x2

    def forward(self, x):
        bsz = x.shape[0]
        # make img into 2 channel -> [B,2,H,W]
        x = x.repeat(1, 2, 1, 1)
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,56]
        x_st1, x_st2 = self.stn(x)
        x_st = torch.cat([x_st1, x_st2], dim=1)  # [B,4,28,56]

        # Perform the usual forward pass
        x_st = F.relu(F.max_pool2d(self.conv1(x_st), 2))                    # [B,10,12,26]
        x_st = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x_st)), 2))   # [B,20, 4, 11]
        x_st = x_st.view(bsz, 880)                                          # [B, 880]
        x_st = F.relu(self.fc1(x_st))                                       # [B, 880] -> [B,50]
        x_st = F.dropout(x_st, training=self.training)

        x_st = self.fc2(x_st)                                               # [B, 50] -> [B, 20]
        x_st = x_st.reshape(-1, 2, 10)                                      # [B,20] -> [B, 2, 10]
        output = F.softmax(x_st, dim=-1)                                    # [B, 2, 10]
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from
        # softargmax
        output = output @ digits                                            # [B, 2, 10] [10] = [B,2]
        return output


class Labels2Out(ExtendedTorchModule):
    def __init__(self, in_features, out_features, learn_labels2out, operation, **kwargs):
        super(Labels2Out, self).__init__('last_layer', **kwargs)

        self.learn_labels2out = learn_labels2out  # if to learn the last layer of to just apply the correct operation
        self.operation = operation

        if learn_labels2out:
            if self.operation == 'add':
                self.fc = torch.nn.Linear(in_features, out_features, bias=False)
            elif self.operation == 'mul':
                self.fc = torch.nn.Parameter(torch.Tensor(out_features, in_features))

    def reset_parameters(self):
        # same init as NMU
        if self.operation == 'mul' and self.learn_labels2out:
            std = math.sqrt(0.25)
            r = min(0.25, math.sqrt(3.0) * std)
            torch.nn.init.uniform_(self.fc, 0.5 - r, 0.5 + r)

    def forward(self, x):
        if not self.learn_labels2out:
            if self.operation == 'add':
                out = x.sum(1)
            elif self.operation == 'mul':
                out = x.prod(1)
        else:
            # x = [B, I=2]
            if self.operation == 'add':
                out = self.fc(x)
            # uses product aggregator
            elif self.operation == 'mul':
                out_size, in_size = self.fc.size()
                x = x.view(x.size()[0], in_size, 1)
                W = self.fc.t().view(1, in_size, out_size)
                # [B,O] = prod([B,I,1], [1,I,O])
                out = torch.prod(x * W, -2)
        # [B, 1]
        return out

    def get_weights(self):
        if self.operation == 'add':
            return self.fc.weight
        elif self.operation == 'mul':
            return self.fc


class Net(ExtendedTorchModule):
    def __init__(self, img2label_in=2, img2label_out=1, use_nalm=False, learn_labels2out=False, operation=None,
                 writer=None, device=None, **kwags):
        super(Net, self).__init__('network', writer=writer, **kwags)
        self.use_nalm = use_nalm
        self.operation = operation
        self.learn_labels2out = learn_labels2out

        # TODO - have way of switching nets automatically
        # self.img2label = Img2LabelsRegression()
        # self.img2label = getattr(torchvision.models, 'resnet18')(num_classes=2)
        # self.img2label = Img2LabelsClassification()
        self.img2label = Img2LabelsIndepImgClf(device=device)
        # self.img2label = Img2LabelsSpatialTransformer(device=device)

        if use_nalm:
            self.labels2Out = \
                ReRegualizedLinearNACLayer(img2label_in, img2label_out, nac_oob='clip',
                                           regualizer_shape='linear', writer=self.writer, **kwags) \
                    if operation == 'add' else \
                    ReRegualizedLinearMNACLayer(img2label_in, img2label_out, nac_oob='clip',
                                                regualizer_shape='linear', writer=self.writer, **kwags)
        else:
            self.labels2Out = Labels2Out(img2label_in, 1, self.learn_labels2out, self.operation,
                                         writer=self.writer, **kwags)

    def forward(self, x):
        # [B, 2]
        z = self.img2label(x)
        out = self.labels2Out(z)
        out = out.squeeze()
        return out, z


class TwoDigitMNISTDataset(Dataset):
    def __init__(self, X, y, z=None, transform=None):
        self.X = X  # [B,H,W]
        self.y = torch.tensor(y).to(torch.float32)  # convert to a float tensor; [B]
        self.z = torch.tensor(z).to(torch.float32)  # [B,2]
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Similar to the MNIST dataset class
        img, target, intermediate_targets = self.X[idx], self.y[idx], self.z[idx]

        if self.transform is not None:
            # [C,H,W] where C is the colour channel
            img = self.transform(self.X[idx])
        return img, target, intermediate_targets


def check_digit_data_split_and_recombine(data):
    """
    Sanity check if the splitting of the 2 digits to single digits (stored across the batch dim) works as expected.
    Also check the recombining of the individual digits back to their two digit single img format.
    """
    print(data.shape)  # [B,C,H,W]
    # splits digits and collapses indep digits into batch dim
    split_digits = torch.cat(data.split(data.shape[-1] // 2, -1), 0)  # [2*B, C=1, 28, 28]
    # recombine the original digits back to the 2-digit img
    recombined_digits = torch.cat(split_digits.split(split_digits.shape[0] // 2, 0), -1)  # [B, C=1, 28, 56]

    # select random img from batch
    img_idx = np.random.randint(0, data.shape[0])

    plt.imshow(data[img_idx].permute(1, 2, 0))
    plt.show()
    plt.imshow(split_digits[img_idx].permute(1, 2, 0))
    plt.show()
    plt.imshow(split_digits[data.shape[0] + img_idx].permute(1, 2, 0))
    plt.show()
    plt.imshow(recombined_digits[img_idx].permute(1, 2, 0))
    plt.show()

class TwoDigitExperiment:
    def __init__(self):
        self._global_step = 0
        self.main()

    def plot_example(self, X, y, z=None):
        """Plot the first 5 images and their labels in a row."""
        if z is None:
            for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, -1), y[:5])):
                plt.subplot(151 + i)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.title(y)
        else:
            for i, (img, y, z) in enumerate(zip(X[:5].reshape(5, 28, -1), y[:5], z[:5])):
                plt.subplot(151 + i)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.title(f"({z[0]},{z[1]}); {y}")
        plt.show()

    def get_label2out_weight(self, model, idx, is_nalm, learn_last_layer):
        # FIXME: generalise
        if is_nalm:
            return model.labels2Out.W.view(-1)[idx].item()
        elif learn_last_layer:
            return model.labels2Out.get_weights().view(-1)[idx].item()

    def calc_reg_loss(self, model, w_scale, args):
        regualizers = model.regualizer()
        reg_loss = regualizers['W'] * w_scale * args.regualizer
        return reg_loss

    def generate_data_samples(self, set_pairs, dataset, x, y, z, args):
        """
        Appends to lists representing the input data and labels.
        Args:
            set_pairs: contains array of strings represetning input digits which are allowed to occur
                e.g. '04' means img 1 = 0 and img 2 = 4
            dataset: mnist dataset (either train or test)
            x: empty list to fill with the input images
            y: empty list to fill with the target labels
            z: empty list to fill with the intermediary image labels e.g. a input with the number 56 will give labels
                '5' and '6'
            args: parser args

        Returns: None. The x,y and z passed in are object references so nothing requires to be returned.

        """
        for train_set_pair in set_pairs:
            for _ in range(args.samples_per_permutation):
                rand_i = np.random.choice(np.where(dataset.targets == int(train_set_pair[0]))[0])
                rand_j = np.random.choice(np.where(dataset.targets == int(train_set_pair[1]))[0])

                temp_image = np.concatenate((dataset.data[rand_i], dataset.data[rand_j]), axis=1)
                x.append(temp_image)
                target_zi = dataset.targets[rand_i]
                target_zj = dataset.targets[rand_j]
                z.append([target_zi, target_zj])

                if args.operation == 'add':
                    y.append(target_zi + target_zj)
                elif args.operation == 'mul':
                    y.append(target_zi * target_zj)
                else:
                    raise KeyError(f'Invalid operation ({args.operation}) given.')

    def eval_dataloader(self, model, device, dataloader, summary_writer, state):
        """
        Prints and logs stats for a given dataloader.
        Model is run in eval mode so weights are fixed.
        """
        model.eval()
        loss = 0
        img_label_losses = torch.zeros(2).to(device)
        correct_output = 0
        correct_img_labels = torch.zeros(2).to(device)
        correct_output_rounded = 0
        correct_img_labels_rounded = torch.zeros(2).to(device)

        with torch.no_grad():
            for data, target, img_label_targets in dataloader:
                data, target, img_label_targets = data.to(device), target.to(device), img_label_targets.to(device)
                ###############################################################################################
                # print(data[0].shape) # [C=1, H=28, W=56])
                # save_image(data[0], '2DMNIST-train_img0-4-1.png') # save img from tensor
                # plt.imshow(data[0].permute(1, 2, 0))  # plot single img
                # plt.show()
                # import sys
                # sys.exit(0)
                ###############################################################################################
                output, img_labels_output = model(data)
                img_label_losses += F.mse_loss(img_labels_output, img_label_targets, reduction='none').sum(dim=0)
                loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

                correct_img_labels += img_labels_output.eq(img_label_targets).sum(dim=0)
                correct_output += output.eq(target.view_as(output)).sum().item()

                correct_img_labels_rounded += img_labels_output.round().eq(img_label_targets).sum(dim=0)
                correct_output_rounded += output.round().eq(target.view_as(output)).sum().item()

        loss /= len(dataloader.dataset)
        img_label_losses /= len(dataloader.dataset)

        acc_output = 100. * correct_output / len(dataloader.dataset)
        acc_label_1 = 100. * correct_img_labels[0].item() / len(dataloader.dataset)
        acc_label_2 = 100. * correct_img_labels[1].item() / len(dataloader.dataset)

        acc_output_rounded = 100. * correct_output_rounded / len(dataloader.dataset)
        acc_label_1_rounded = 100. * correct_img_labels_rounded[0].item() / len(dataloader.dataset)
        acc_label_2_rounded = 100. * correct_img_labels_rounded[1].item() / len(dataloader.dataset)

        # prints the average: epoch loss, accuracy of the final output and intermediate label losses.
        print('{}: {:.5f}, acc: {}/{} ({:.2f}%), img1: {:.5f} ({:.2f}%), img2: {:.5f} ({:.2f}%)\t '
              'Rounded: acc: {}/{} ({:.2f}%), img1:({:.2f}%), img2: ({:.2f}%)'.format(
            state, loss, correct_output, len(dataloader.dataset),
            acc_output,
            img_label_losses[0],
            acc_label_1,
            img_label_losses[1],
            acc_label_2,
            correct_output_rounded, len(dataloader.dataset),
            acc_output_rounded,
            acc_label_1_rounded,
            acc_label_2_rounded
        ))

        # log stats to tensorboard
        summary_writer.add_scalar(f'metric/{state}/output/mse', loss)
        summary_writer.add_scalar(f'metric/{state}/label1/mse', img_label_losses[0])
        summary_writer.add_scalar(f'metric/{state}/label2/mse', img_label_losses[1])

        summary_writer.add_scalar(f'metric/{state}/output/acc', acc_output)
        summary_writer.add_scalar(f'metric/{state}/label1/acc', acc_label_1)
        summary_writer.add_scalar(f'metric/{state}/label2/acc', acc_label_2)

        summary_writer.add_scalar(f'metric/{state}/output_rounded/acc', acc_output_rounded)
        summary_writer.add_scalar(f'metric/{state}/label1_rounded/acc', acc_label_1_rounded)
        summary_writer.add_scalar(f'metric/{state}/label2_rounded/acc', acc_label_2_rounded)

        model.train()

    def epoch_step(self, model, train_loader, args, optimizer, epoch, w_scale, summary_writer, test_loader):
        """
        Train and test the model for a single epoch. Logging occurs at the start of the epoch before any optimisation
        meaning epoch 0 will log the model stats before any param updates.
        """
        model.train()
        # num_samples_processed = 0
        # epoch_loss = 0
        # epoch_label_losses = torch.zeros(2)

        for batch_idx, (data, target, img_label_targets) in enumerate(train_loader):
            data, target, img_label_targets = data.to(args.device), target.to(args.device), img_label_targets.to(args.device)
            self._global_step += 1
            summary_writer.set_iteration(self._global_step)
            summary_writer.add_scalar('epoch', epoch)

            # log to tensorboard. Metrics include stats for training, testing (over the entire dataloader) and the
            # weights or the label2out layer (if any exists).
            if epoch % args.log_interval == 0 and batch_idx == 0:
                self.eval_dataloader(model, args.device, train_loader, summary_writer, 'train')
                self.eval_dataloader(model, args.device, test_loader, summary_writer, 'test')
                if args.learn_labels2out:
                    # plot weight values of final layer
                    summary_writer.add_scalar('label2out/weights/w0',
                                              self.get_label2out_weight(model, 0, args.use_nalm, args.learn_labels2out))
                    summary_writer.add_scalar('label2out/weights/w1',
                                              self.get_label2out_weight(model, 1, args.use_nalm, args.learn_labels2out))

                # if args.use_nalm:
                #     print(model.labels2Out.W)
                # if not args.use_nalm and args.learn_labels2out:
                #     print(model.labels2Out.get_weights())

            optimizer.zero_grad()

            with summary_writer.force_logging(epoch % args.log_interval == 0 and batch_idx == 0):
                # check_digit_data_split_and_recombine(data)
                output, img_labels_output = model(data)

            loss = F.mse_loss(output, target, reduction='sum')
            total_loss = loss

            # mse on [B,2] -> [1,2]
            # img_label_losses = F.mse_loss(img_labels_output, img_label_targets, reduction='none').sum(dim=0)
            # epoch_label_losses += img_label_losses
            # epoch_loss += loss

            # img1_loss = img_label_losses[0] / len(target)
            # img2_loss = img_label_losses[1] / len(target)

            loss /= len(target)

            if args.use_nalm:
                #args.regualizer = loss.item()   # make nalm reg scale to the loss magnitude
                reg_loss = self.calc_reg_loss(model, w_scale, args)
                total_loss += reg_loss

            if total_loss.requires_grad:
                total_loss.backward()

            optimizer.step()
            model.optimize(loss)

            # print minibatch train stats
            # num_samples_processed += len(data)
            # if batch_idx % args.mb_log_interval == 0:
            #     print('Train: [{}/{} ({:.0f}%)], mb: {:.6f}, reg: {:.6f}, img1 {:.6f}, img2 {:.6f}'.format(
            #         num_samples_processed, len(train_loader.dataset),
            #         100. * (batch_idx + 1) / len(train_loader),
            #         loss.item(), reg_loss, img1_loss, img2_loss
            #     ))

        # print epoch stats averaged over minibatches
        # epoch_loss /= len(train_loader.dataset)
        # epoch_label_losses /= len(train_loader.dataset)
        # print('Train: {:.5f}, img1 {:.5f}, img2 {:.5f}'.format(epoch_loss, epoch_label_losses[0], epoch_label_losses[1]))

    def reset_weights(self, m):
        """
          Reset model weights to avoid weight leakage.
        """
        if hasattr(m, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {m}')
            m.reset_parameters()

    def main(self):
        ####################################################################################################################
        args = parser.parse_args()
        args.device = torch.device("cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda")

        ####################################################################################################################
        def set_reproducability_flags(seed):
            # Set reproducability flags - see https://pytorch.org/docs/stable/notes/randomness.html
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np_rng = np.random.RandomState(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.set_default_dtype(torch.float32)
            return np_rng

        fold_idx = args.seed
        set_reproducability_flags(fold_idx)
        assert fold_idx < args.num_folds, "Seed must be less than number of folds. (Seed is analagous to fold idx.)"

        # def seed_worker(worker_id):
        #     worker_seed = args.seed
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)

        ####################################################################################################################
        # TODO - only for dev - remove when real run.
        use_dummy_writer = False  # dummywriter won't save tensorboard files
        
        ################################################################################
        # check unsupported edge case
        assert (not args.use_nalm or args.learn_labels2out), "NALM with fixed weights is not supported"

        # print parser args
        print(' '.join(f'{k}: {v}\n' for k, v in vars(args).items()))

        ################################################################################
        # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/9
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if args.rgb else NoneTransform(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # we will be creating a new dataset which joins 2 mnist images together. As the __get__item applies the Transform
        # we cannot use it here as it won't get called.
        train_dataset = datasets.MNIST(args.data_path, train=True, download=True)
        test_dataset = datasets.MNIST(args.data_path, train=False, download=False)
        ################################################################################

        unique_pairs = [str(x) + str(y) for x in range(10) for y in range(10)]
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=np.random.RandomState(0))
        unique_pairs_np = np.asarray(unique_pairs)

        # todo - only use if run only does 1 fold
        fold_unique_pairs_list = list(kf.split(unique_pairs))
        train_index, test_index = fold_unique_pairs_list[fold_idx]

        ################################################################################
        # todo: use loop if want to do all folds in sequence
        # for fold_idx, (train_index, test_index) in enumerate(kf.split(unique_pairs)):
        # set seed for each new fold. Makes reproducability simpler if rereun/checkpointing of a single fold is required.
        np_rng = set_reproducability_flags(fold_idx)
        self._global_step = 0   # reset iteration step count every new fold

        print('=====================================================')
        print(f"Fold {fold_idx}")
        summary_writer = create_tb_writer(args, fold_idx, use_dummy=use_dummy_writer)
        print("Writer name: ", summary_writer.name)
        print('-----------------------------------------------------')
        # get the digit pair that are used for the current fold
        test_set_pairs = unique_pairs_np[test_index]
        train_set_pairs = unique_pairs_np[train_index]

        # Sanity checks
        assert (len(test_set_pairs) == 10)
        assert (len(train_set_pairs) == 90)
        for test_set_pair in test_set_pairs:
            assert (test_set_pair not in train_set_pairs)

        ################################################################################
        X_train = []
        y_train = []
        z_train = [] # img labels
        self.generate_data_samples(train_set_pairs, train_dataset, X_train, y_train, z_train, args)

        X_test = []
        y_test = []
        z_test = []
        self.generate_data_samples(test_set_pairs, test_dataset, X_test, y_test, z_test, args)

        ################################################################################

        X_train_shuffled, y_train_shuffled, z_train_shuffled = \
            shuffle(X_train, y_train, z_train, random_state=np_rng)
        X_test_shuffled, y_test_shuffled, z_test_shuffled = \
            shuffle(X_test, y_test, z_test, random_state=np_rng)

        # can't convert to a tensor yet otherwise the Dataset Transforms won't work
        two_digit_train_X = np.asarray(X_train_shuffled)
        two_digit_train_y = np.asarray(y_train_shuffled)
        two_digit_train_z = np.asarray(z_train_shuffled)
        two_digit_test_X = np.asarray(X_test_shuffled)
        two_digit_test_y = np.asarray(y_test_shuffled)
        two_digit_test_z = np.asarray(z_test_shuffled)

        processed_train_dataset = TwoDigitMNISTDataset(two_digit_train_X, two_digit_train_y, two_digit_train_z, transform)
        processed_test_dataset = TwoDigitMNISTDataset(two_digit_test_X, two_digit_test_y, two_digit_test_z, transform)

        # FIXME -> worker inits
        train_dataloader = DataLoader(processed_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.dataset_workers,
                                      worker_init_fn=None, pin_memory=True)
        test_dataloader = DataLoader(processed_test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.dataset_workers,
                                     worker_init_fn=None, pin_memory=True)

        ############################################################################################################
        # Sanity check the generated dataset, displaying the first 5 samples for both train and test datasets
        # self.plot_example(two_digit_train_X, two_digit_train_y, two_digit_train_z)
        # self.plot_example(two_digit_test_X, two_digit_test_y, two_digit_test_z)

        ############################################################################################################

        # create model and optimizer
        model = Net(
                    writer=summary_writer.every(args.log_interval).verbose(args.verbose),
                    img2label_in=2, img2label_out=1,
                    use_nalm=args.use_nalm,
                    learn_labels2out=args.learn_labels2out,
                    operation=args.operation,
                    beta_nau=args.beta_nau,
                    nau_noise=args.nau_noise,
                    nmu_noise=args.nmu_noise,
                    noise_range=args.noise_range,
                    device=args.device
                ).to(args.device)
        model.apply(self.reset_weights)

        ###############################################################################################################
        # PRETRAINED MODEL VISUALISATION CODE
        # # quick and dirty loading of pretrained model (no setting random states/ opt)
        # # TODO - make sure the correct flags have been set for the model. ESP the SEED/FOLD val!
        # # load pretrained model
        # # load_filename = '22_f2_op-mul_nalmT_learnLT_s2'
        # # load_filename = '23_f2_op-mul_nalmF_learnLF_s2'
        # # load_filename = '24_f2_op-mul_nalmT_learnLT_s2'
        # # load_filename = '25_f2_op-mul_nalmF_learnLT_s2'
        # load_filename = '70_f2_op-mul_nalmT_learnLT_s2'
        # checkpoint = torch.load(f'../save/{load_filename}.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        # print('Pretrained model loaded')
        # model.to(args.device)
        # model.eval()
        #
        # from experiments.ST_mnist_labels import digit_confusion_matrix
        # digit_confusion_matrix(model.img2label, test_dataloader, args.device, digit_idx=None, round=False, old_model=True,
        #                        save_data={"model_name": load_filename,
        #                                   "save_dir": "../save/two_digit_mnist_plots/"})
        #
        # import sys
        # sys.exit()
        ###############################################################################################################

        if fold_idx == 0:
            print(model)
            print(f"Param count (all): {sum(p.numel() for p in model.parameters())}")
            print(f"Param count (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

        # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate, rho=0.95) #torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # see https://github.com/Coderx7/SimpleNet_Pytorch/blob/master/main.py
        #optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, rho=0.9, eps=1e-3, weight_decay=0.001)
        milestones = [10, 20, 50]
        #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

        ###########################################################################################################
        # Train/test loop
        for epoch in range(args.max_epochs + 1):
            print(f"Epoch {epoch}")
            w_scale = max(0, min(1, (
                    (epoch - args.regualizer_scaling_start) /
                    (args.regualizer_scaling_end - args.regualizer_scaling_start)
            )))

            self.epoch_step(model, train_dataloader, args, optimizer, epoch, w_scale, summary_writer, test_dataloader)
            #current_learning_rate = float(scheduler.get_last_lr()[-1])
            #print('lr:', current_learning_rate)
            #scheduler.step()
        ###########################################################################################################

        if not use_dummy_writer:
            summary_writer._root.close()

        if not args.no_save:
            writer.save_model_checkpoint(summary_writer.name, epoch + 1, model, optimizer,
                                         {'torch': torch.get_rng_state(), 'numpy': np.random.get_state()},
                                         args=args)
            print(f'model saved for fold {fold_idx}')


if __name__ == '__main__':
    TwoDigitExperiment()


    # args = parser.parse_args()
    # args.use_nalm = True
    # args.learn_labels2out = False
    # args.operation = 'add'
    # summary_writer = create_tb_writer(args, -1, use_dummy=True)
    # model = Net(
    #     writer=summary_writer.every(args.mb_log_interval).verbose(args.verbose),
    #     img2label_in=2, img2label_out=1,
    #     use_nalm=args.use_nalm,
    #     learn_labels2out=args.learn_labels2out,
    #     operation=args.operation,
    #     beta_nau=args.beta_nau,
    #     nau_noise=args.nau_noise,
    #     nmu_noise=args.nmu_noise,
    #     noise_range=args.noise_range
    # )
    # checkpoint = torch.load('../save/test.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
