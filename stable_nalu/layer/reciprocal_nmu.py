import math

import torch

from ._abstract_recurrent_cell import AbstractRecurrentCell
from ..abstract import ExtendedTorchModule
from ..functional import mnac, Regualizer, RegualizerNAUZ, sparsity_error
from torch.distributions.beta import Beta


class ReciprocalNMULayer(ExtendedTorchModule):
    """Implements the NRU (Neural Reciprocal Unit)
        Use in SLTR: --layer-type NRU and all other args are same as using NMU (i.e. ReRegualizedLinearMNAC)
        Use in ADT: TODO

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 mnac_epsilon=0, regualizer_z=0,
                 **kwargs):
        super().__init__('nru', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_epsilon = mnac_epsilon
        self.nac_oob = nac_oob
        self.use_beta_init = kwargs['beta_nau']
        self.div_mode = kwargs['nru_div_mode']

        # Use NAU type reg for sparsity
        self._regualizer_bias = Regualizer(
            support='nac', type='bias',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon
        )
        # Use NAU type reg for sparsity
        self._regualizer_oob = Regualizer(
            support='nac', type='oob',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon,
            zero=self.nac_oob == 'clip'
        )
        # Want nau not nmu because weights can include negatives
        self._regualizer_nau_z = RegualizerNAUZ(
            zero=regualizer_z == 0
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

        self.use_trash_cell = kwargs['trash_cell']
        if self.use_trash_cell:
            self.trash_cell = torch.nn.Parameter(torch.Tensor(1))

    def reset_parameters(self):
        # NAU init
        if self.use_beta_init:
            self.W.data = (Beta(torch.tensor([7.]), torch.tensor([7.])).sample(self.W.shape).squeeze(
                -1) * 2) - 1  # sample in range [-1,1]
            # self.W.data = (Normal(torch.tensor([0.5]), torch.tensor([math.sqrt(1/60)])).sample(self.W.shape).squeeze(-1)*2)-1    # sample in range [-1,1]
        else:
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
            r = min(0.5, math.sqrt(3.0) * std)
            torch.nn.init.uniform_(self.W, -r, r)

        if self.use_trash_cell:
            torch.nn.init.zeros_(self.trash_cell)
        # self.W = torch.nn.Parameter(torch.Tensor([[1., -1.]]))  # TODO - used fixed NMU
        # self.W.requires_grad = False

        self._regualizer_nau_z.reset()

    def optimize(self, loss):
        self._regualizer_nau_z.reset()

        # clip [-1,1]
        if self.nac_oob == 'clip':
            self.W.data.clamp_(-1.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nau_z(self.W),
            'W-OOB': self._regualizer_oob(self.W)   # TODO: remove?
        })

    def forward(self, x, reuse=False):
        if self.allow_random:
            self._regualizer_nau_z.append_input(x)

        # [-1,1] clamping if using OOB regularisation
        W = torch.clamp(self.W, -1.0 + self.mnac_epsilon, 1.0) \
            if self.nac_oob == 'regualized' \
            else self.W

        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        out = mnac(x, W, mode=self.div_mode, is_training=self.training)

        if self.use_trash_cell and self.training:
            out = out + self.trash_cell

        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class ReciprocalNMUCell(AbstractRecurrentCell):
    """Implements the NRU (Neural Reciprocal Unit) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ReciprocalNMULayer, input_size, hidden_size, **kwargs)
