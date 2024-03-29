
import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac, Regualizer, RegualizerNMUZ, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell

class ReRegualizedLinearMNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 mnac_epsilon=0, mnac_normalized=False, regualizer_z=0,
                 **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized
        self.mnac_epsilon = mnac_epsilon
        self.nac_oob = nac_oob

        self._regualizer_bias = Regualizer(
            support='mnac', type='bias',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon
        )
        self._regualizer_oob = Regualizer(
            support='mnac', type='oob',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon,
            zero=self.nac_oob == 'clip'
        )
        self._regualizer_nmu_z = RegualizerNMUZ(
            zero=regualizer_z == 0
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)
        self.use_noise = kwargs['nmu_noise']
        self.noise_range = kwargs['noise_range']

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

        self._regualizer_nmu_z.reset()

    def optimize(self, loss):
        self._regualizer_nmu_z.reset()

        if self.nac_oob == 'clip':
            self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nmu_z(self.W),
            'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x, reuse=False):
        if self.use_noise and self.training:
            # check if to use the batch stats for the upper bound (i.e. if range is [1,0]
            if self.noise_range == [1, 0]:
                x_sdev = x.std()
                noise_upper = (1 + (1/x_sdev)).item()
                self.writer.add_scalar('snmu_noise/upper', noise_upper, verbose_only=False)
                noise = torch.Tensor(x.shape).uniform_(1, noise_upper).to(self.W.device)
            # use noise_range arg for lower and upper bounds
            else:
                noise = torch.Tensor(x.shape).uniform_(self.noise_range[0], self.noise_range[1]).to(self.W.device)  # [B,I]
            x *= noise
            
        if self.allow_random:
            self._regualizer_nmu_z.append_input(x)
    
        W = torch.clamp(self.W, 0.0 + self.mnac_epsilon, 1.0) \
            if self.nac_oob == 'regualized' \
            else self.W

        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W, verbose_only=False if self.use_robustness_exp_logging else True)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=self.use_robustness_exp_logging)


        if self.mnac_normalized:
            c = torch.std(x)
            x_normalized = x / c
            z_normalized = mnac(x_normalized, W, mode='prod')
            out = z_normalized * (c ** torch.sum(W, 1))
        else:
            out = mnac(x, W, mode='prod')

        # apply denoising if sNMU is used
        if self.use_noise and self.training:
            # [B,O] / mnac([B,I], [O,I] 'prod') --> [B,O] / [B,O] --> [B,O]
            out = out / mnac(noise, W, mode='prod')
            
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class ReRegualizedLinearMNACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ReRegualizedLinearMNACLayer, input_size, hidden_size, **kwargs)
