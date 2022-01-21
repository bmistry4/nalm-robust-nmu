# encoding: utf-8
"""
This class is copied from the repo: https://github.com/WarBean/tps_stn_pytorch.
Modifications have been made to be used with this work.

Spatial Transformer using Thin Plate Splines for localisation
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tps_grid_gen import TPSGridGen
from stable_nalu.abstract import ExtendedTorchModule

# what the other affine ST nets have been using
class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        # Spatial transformer localization-network
        self.loc = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        self.fc1 = nn.Linear(320, 32)
        self.fc2 = nn.Linear(32, num_output)

    def forward(self, x):
        # x = [B, 2, 42, 42] -> xs1 = [B,20, 4,4]
        x = self.loc(x)
        # [B, 320]
        x = x.view(-1, 20 * 4 * 4)
        # [B, 2*H*W]
        x = self.fc2(F.relu(self.fc1(x)))
        return x

# fixme - this is the way they do it
# class CNN(nn.Module):
#     def __init__(self, num_output):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(20*7*7, 50)
#         self.fc2 = nn.Linear(50, num_output)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 20*7*7)    # fixme - changed to work for the [42,42] img size
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x


class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x))


class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = torch.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)


class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)


class STN_TPS(nn.Module):

    def __init__(self, image_height, image_width,
                 span_range_height=0.9, span_range_width=0.9,
                 grid_height_points=4, grid_width_points=4,
                 unbounded_stn=False
                 ):
        super(STN_TPS, self).__init__()
        self.unbounded_stn = unbounded_stn
        self.image_height, self.image_width = image_height, image_width
        # number of control/landmark points will be grid_height * grid_width
        grid_height, grid_width = grid_height_points, grid_width_points

        r1 = span_range_height
        r2 = span_range_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        # create an evenly spaced out grid of the base control points (normalized between (-1,1)).
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)    # [K,2]

        # create localisation nets which will creates the control points on the original image.
        # require 2 loc nets; 1 for each digit to find.
        if self.unbounded_stn:
            self.loc1 = UnBoundedGridLocNet(grid_height, grid_width, target_control_points)
            self.loc2 = UnBoundedGridLocNet(grid_height, grid_width, target_control_points)
        else:
            self.loc1 =BoundedGridLocNet(grid_height, grid_width, target_control_points)
            self.loc2 = BoundedGridLocNet(grid_height, grid_width, target_control_points)
        # will apply the thin plate spline transformation to the org img control points
        self.tps = TPSGridGen(self.image_height, self.image_width, target_control_points)

    def forward(self, x):
        # x = [B, 2, 42, 42]

        bsz = x.size(0)

        # [B, K, 2]
        source_control_points_1 = self.loc1(x)
        # [B, HW, 2]
        source_coordinate_1 = self.tps(source_control_points_1)
        # [B,H,W,2]
        grid_1 = source_coordinate_1.view(bsz, self.image_height, self.image_width, 2)
        # [B, C=2, H=42, W=42] = grid_sample([B, 2, H, W], [B,H,W,2])
        transformed_x1 = F.grid_sample(x, grid_1, align_corners=False)

        # apply the ST-TPS, localising on the second digit.
        source_control_points_2 = self.loc2(x)
        source_coordinate_2 = self.tps(source_control_points_2)
        grid = source_coordinate_2.view(bsz, self.image_height, self.image_width, 2)
        transformed_x2 = F.grid_sample(x, grid, align_corners=False)

        return transformed_x1, transformed_x2


class Img2LabelsSpatialTransformerTPSConvNoConcat(ExtendedTorchModule):
    """
    Recreating the MNIST addition network from the Spatial Transformers Paper BUT
    with a label clf after the STNs (using a non-concat method)
    """

    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformerTPSConvNoConcat, self).__init__('stn_clf', **kwargs)
        self.device = device
        image_height = kwargs['image_height']
        image_width = kwargs['image_width']
        span_range_height = kwargs['span_range_height']
        span_range_width = kwargs['span_range_width']
        grid_height_points = kwargs['grid_height_points']
        grid_width_points = kwargs['grid_width_points']
        unbounded_stn = kwargs['tps_unbounded_stn']

        # todo try version which uses CNN to match their code
        # classifier once ST output is given
        self.st_fcn = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(4096, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        # Spatial transformer network
        self.stn = STN_TPS(image_height, image_width, span_range_height, span_range_width,
                           grid_height_points, grid_width_points, unbounded_stn)

    def forward(self, x):
        # x = [B, 2, 42, 42]
        # transform the input where each return tensor has shape [B,2,H,W]
        x_st1, x_st2 = self.stn(x)
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from

        # predict digit 1
        z_st1 = F.avg_pool2d(x_st1, kernel_size=2)
        z_st1 = self.st_fcn(z_st1)
        z_st1 = F.softmax(z_st1, dim=-1)
        z_st1 = z_st1 @ digits  # [B, 2, 10] [10] = [B,1]

        # predict digit 2
        z_st2 = F.avg_pool2d(x_st2, kernel_size=2)
        z_st2 = self.st_fcn(z_st2)
        z_st2 = F.softmax(z_st2, dim=-1)
        z_st2 = z_st2 @ digits  # [B, 2, 10] [10] = [B,1]

        # [B,2]
        output = torch.stack((z_st1, z_st2), dim=1)
        return output, x_st1, x_st2
