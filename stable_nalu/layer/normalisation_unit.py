import torch
from torch.nn.functional import relu
from ..abstract import ExtendedTorchModule


class NNULayer(ExtendedTorchModule):
    """Implements the Neural (De)Normalisation Unit i.e. NNU

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('nnu', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        self.fc = torch.nn.Linear(self.in_features, self.out_features, bias=False)

    # def reset_parameters(self):
    #     self.fc.weight.data.fill_(1.)
        # self.fc.bias.data.fill_(0.)

    # def optimize(self, loss):
    #     # called after weight update - opt.step()
    #     pass

    # def regualizer(self):
    #     return super().regualizer({})

    def forward(self, inputs: tuple):
        current_x, inital_x = inputs
        fc_result = self.fc(inital_x)
        summative_constants = fc_result[:, 0:self.out_features // 2]
        multiplicative_constants = fc_result[:, self.out_features // 2:]  # only allow positive constants
        out = (current_x + summative_constants) * multiplicative_constants
        # out = (current_x * multiplicative_constants) + summative_constants
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
