import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac, Regualizer, RegualizerNMUZ, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell


class ConcatReciprocalNMUGatedSignRetrievalLayer(ExtendedTorchModule):
    """Implements the NMRU with separate sign retrieval using the Real NPU method

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 mnac_epsilon=0, mnac_normalized=False, regualizer_z=0,
                 **kwargs):
        super().__init__('gated-nmru-sign', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized
        self.mnac_epsilon = mnac_epsilon
        self.nac_oob = nac_oob
        self.eps = torch.finfo(torch.float).eps  # 32-bit eps

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
        if kwargs['regualizer_gate']:
            self._regualizer_g = Regualizer(
                support='mnac', type='bias',
                shape='linear'
            )
        else:
            self._regualizer_g = Regualizer(zero=True)

        self.W = torch.nn.Parameter(torch.Tensor(out_features, 2*in_features))  # [O, 2I]
        self.g = torch.nn.Parameter(torch.Tensor(2*in_features))
        self.register_parameter('bias', None)
        self.use_noise = kwargs['nmu_noise']

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

        torch.nn.init.ones_(self.g)
        self.g.data /= 2.0

        self._regualizer_nmu_z.reset()

    def optimize(self, loss):
        self._regualizer_nmu_z.reset()

        if self.nac_oob == 'clip':
            self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)
            self.g.data.clamp_(0.0, 1.0)

    def regualizer(self):
        return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nmu_z(self.W),
            'W-OOB': self._regualizer_oob(self.W),
            'g-NPU': self._regualizer_g(self.g)
        })

    def concat_input_reciprocal(self, x):
        reciprocal = (x + self.eps).reciprocal() if self.training else x.reciprocal()
        return torch.cat((x, reciprocal), 1)    # concat on input dim

    def forward(self, x):
        # Concat the reciprocal of the input to the original input
        x = self.concat_input_reciprocal(x)
        # apply gating
        x = x * self.g
        # Create 'negated sign matrix' where Let all negative values become pi, all other values are set to 0
        k = torch.max(-torch.sign(x), torch.zeros_like(x)) * math.pi
        # will only be calculating the magnitude based on the absolute value of the input
        x = torch.abs(x)

        if self.use_noise and self.training:
            noise = torch.Tensor(x.shape).uniform_(1, 5).to(self.W.device)  # [B,I]
            x *= noise

        if self.allow_random:
            self._regualizer_nmu_z.append_input(x)

        W = torch.clamp(self.W, 0.0 + self.mnac_epsilon, 1.0) \
            if self.nac_oob == 'regualized' \
            else self.W

        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        self.writer.add_histogram('gate', self.g)
        self.writer.add_scalar('gate/sparsity_error', sparsity_error(self.g), verbose_only=False)
        self.writer.add_tensor('gate', self.g)

        magnitude = mnac(x, W, mode='prod')

        # apply denoising if sNMU is used
        if self.use_noise and self.training:
            # [B,O] / mnac([B,I], [O,I] 'prod') --> [B,O] / [B,O] --> [B,O]
            magnitude = magnitude / mnac(noise, W, mode='prod')

        # apply sign retrieval
        out = magnitude * torch.cos(k.matmul(self.W.T))
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class ConcatReciprocalNMUGatedSignRetrievalCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ConcatReciprocalNMUGatedSignRetrievalLayer, input_size, hidden_size, **kwargs)
