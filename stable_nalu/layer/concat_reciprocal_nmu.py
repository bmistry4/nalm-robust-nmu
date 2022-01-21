import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac, Regualizer, RegualizerNMUZ, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell


class ConcatReciprocalNMULayer(ExtendedTorchModule):
    """Implements the NMRU

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 mnac_epsilon=0, mnac_normalized=False, regualizer_z=0,
                 **kwargs):
        super().__init__('nmru', **kwargs)
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

        self.W = torch.nn.Parameter(torch.Tensor(out_features, 2*in_features))  # [O, 2I]
        self.register_parameter('bias', None)
        self.use_noise = kwargs['nmu_noise']

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

        # self.W = torch.nn.Parameter(torch.Tensor([[1, 0., 0, 1]]))  # TODO - used fixed NMU
        # self.W.requires_grad = False

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

    def concat_input_reciprocal(self, x):
        reciprocal = (x + self.eps).reciprocal() if self.training else x.reciprocal()
        return torch.cat((x, reciprocal), 1)    # concat on input dim

    def forward(self, x):
        # Concat the reciprocal of the input to the original input
        x = self.concat_input_reciprocal(x)

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


class ConcatReciprocalNMUCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ConcatReciprocalNMULayer, input_size, hidden_size, **kwargs)
