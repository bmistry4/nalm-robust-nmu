# encoding: utf-8
"""
This class is copied from the repo: https://github.com/WarBean/tps_stn_pytorch
"""
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable


def compute_partial_repr(input_points, control_points):
    """
    Calculates phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2 between 2 sets of points.
    Used as the RBF distance measure.
    Args:
        input_points: coords to calc pairswise dist between
        control_points: landmark points (coords)

    Returns: distance matrix

    """
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGen(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        """

        Args:
            target_height: height of the resulting image
            target_width: width of the resulting image
            target_control_points: control points for the target image
        """
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        ###############################################################################################################
        # create padded kernel matrix -> [R 1 C' ; 1 0 0 ; C'^T 0 0] of shape [N+3, N+3]
        # R = distance matrix between the target control points
        forward_kernel = torch.zeros(N + 3, N + 3)
        # [N, N] -> diagonal will be 0's
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        # 1
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        # C' and C'^T
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)
        ###############################################################################################################

        # create target coordinate matrix
        HW = target_height * target_width
        # target image pixel coords (between [-1,1])
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        # pairwise distances between the image pxl coords and the control point coords for the target image
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points) # [HW, K]
        # representing a vector similar to hat(p') from eq.3 in the paper https://arxiv.org/pdf/1603.03915.pdf
        # [HW, K+3] ; the +3 reps the [1 x_i y_i]
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)
        ###############################################################################################################

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        # reps V from http://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf
        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        # TPS params -> includes the affine parans, coord weights and RBF kernel weights i.e. (a, ax_i, ay_i, w_i)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        # apply the TPS to the trgs coords to get the src (org) img coords
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate
