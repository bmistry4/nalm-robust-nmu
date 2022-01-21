import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, BasicLayer


class SingleLayerNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=2, hidden_size=2, output_size=1, writer=None, nac_mul='none', eps=1e-7, mlp_depth=0, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name  # layer_type arg
        self.input_size = input_size
        self.nac_mul = nac_mul
        self.eps = eps
        self.use_mlp = unit_name in BasicLayer.ACTIVATIONS

        # special case: NAC* requires a NAC+ as it's layer. The operation is dealt with in the forward pass of this class
        if unit_name == 'MNAC':
            unit_name = 'NAC'

        # self.norm_unit = GeneralizedLayer(input_size, input_size*2, 'NNU', writer=self.writer, name='norm_layer')

        hidden_layers_list = [GeneralizedLayer(hidden_size,
                                        hidden_size,
                                        unit_name,
                                        writer=self.writer,
                                        name=f'layer_{i}{i+1}',
                                        eps=eps, **kwags) for i in range(mlp_depth)]
        self.layer_1 = torch.nn.Sequential(
            GeneralizedLayer(input_size,
                hidden_size if unit_name in BasicLayer.ACTIVATIONS else output_size,
                unit_name,
                writer=self.writer,
                name='layer_1',
                eps=eps, **kwags),
            *hidden_layers_list)
        # self.denorm_unit = GeneralizedLayer(input_size, output_size*2, 'NNU', writer=self.writer, name='norm_layer')

        # # create a 1-hidden layer MLP
        if self.use_mlp:
            self.layer_2 = GeneralizedLayer(hidden_size, output_size,
                                        'linear',
                                        writer=self.writer,
                                        name='layer_2',
                                        eps=eps, **kwags)

        self.reset_parameters()
        self.z_1_stored = None

    def reset_parameters(self):
        if isinstance(self.layer_1, torch.nn.Sequential):
            for c in self.layer_1.children():
                c.reset_parameters()
        else:
            self.layer_1.reset_parameters()
        if self.use_mlp:
            self.layer_2.reset_parameters()

    def regualizer(self):
        if self.nac_mul == 'max-safe':
            return super().regualizer({
                'z': torch.mean(torch.relu(1 - self.z_1_stored))
            })
        else:
            return super().regualizer()

    # def normalise(self, input):
    #     return input / input.abs().sum(dim=1).unsqueeze(1)

    def forward(self, input):
        self.writer.add_summary('x', input)
        # inital_input = input
        # input = self.norm_unit((input, input))
        # input = self.normalise(inital_input)

        # do mulitplicative path (NAC*)
        if self.unit_name == 'MNAC':
            z_1 = torch.exp(self.layer_1(torch.log(torch.abs(input) + self.eps)))
        else:
            z_1 = self.layer_1(input)

        self.z_1_stored = z_1
        self.writer.add_summary('z_1', z_1)

        # mlp has 2 layers
        if self.use_mlp:
            z_2 = self.layer_2(z_1)
            self.writer.add_summary('z_2', z_2)
            return z_2

        # z_1 = self.denorm_unit((z_1, inital_input))

        return z_1

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
