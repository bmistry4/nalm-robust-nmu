
import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import Regualizer, RegualizerNAUZ, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell

class MultiWeightNAULayer(ExtendedTorchModule):
    # NAU with relaxed weight sharing - influence from https://arxiv.org/pdf/1511.08228.pdf (Neural GPU)
    """Implements the RegualizedLinearNAC

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared', regualizer_z=0,
                 **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.nac_oob = nac_oob
        self.use_noise = kwargs['nau_noise']

        self._regualizer_bias = Regualizer(
            support='nac', type='bias',
            shape=regualizer_shape
        )
        self._regualizer_oob = Regualizer(
            support='nac', type='oob',
            shape=regualizer_shape,
            zero=self.nac_oob == 'clip'
        )
        self._regualizer_nau_z = RegualizerNAUZ(
            zero=regualizer_z == 0
        )
        self._regualizer_w_set = Regualizer(
            support='nac', type='wset',
            shape='linear', 
            zero=False
        )

        self.register_parameter('bias', None)
        self.Ws = [] 
        for i in range(kwargs['W_set_count']):
          self.register_parameter('W_'+str(i), torch.nn.Parameter(torch.Tensor(out_features, in_features)))
          self.Ws.append(getattr(self, 'W_'+str(i)))
    
    def get_W_idx(self):
        return self.writer.get_iteration() % len(self.Ws)
    
    def get_Wi(self):
      return self.Ws[self.get_W_idx()]
      
    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        
        for W in self.Ws:
          torch.nn.init.uniform_(W, -r, r)

    def optimize(self, loss):
        self._regualizer_nau_z.reset()

        if self.nac_oob == 'clip':
            self.get_Wi().data.clamp_(-1.0, 1.0)
    
    def remove_param(self, name):
        del self._parameters[name]
        
    def average_Ws(self):
      avg = torch.nn.Parameter(torch.mean(torch.stack(self.Ws), 0).detach())
      
      for i in range(len(self.Ws)):
        self.Ws[0].requires_grad = False
        self.remove_param('W_'+str(i))
        # remove as go through list, so only need to remove 1st 
        del self.Ws[0]
      self.register_parameter('W_avg', avg)
      self.Ws = [avg]
      return avg

    def regualizer(self):
         return super().regualizer({
            'W': self._regualizer_bias(self.get_Wi()),
            'z': self._regualizer_nau_z(self.get_Wi()),
            'W-OOB': self._regualizer_oob(self.get_Wi()),
            'W-set': self._regualizer_w_set(self.Ws)
        })

    def forward(self, x, reuse=False):
        if self.use_noise and self.training:
            a = 1/x.var()
            
            # additive noise - unique f.e. element 
            noise = torch.Tensor(x.shape).uniform_(-a, a)  # [B,I]
            x += noise
            
            # multiplicative noise - unique f.e. b.item
            #noise = torch.Tensor(x.shape[0],1).uniform_(-a, a)  # [B,1]
            #x *= noise
            
        if self.allow_random:
            self._regualizer_nau_z.append_input(x)

        W = torch.clamp(self.get_Wi(), -1.0, 1.0)
        self.writer.add_histogram('W', self.get_Wi())
        self.writer.add_tensor('W', self.get_Wi())
        self.writer.add_scalar('W/sparsity_error', sparsity_error(self.get_Wi()), verbose_only=False)
        
        out = torch.nn.functional.linear(x, W, self.bias)
        
        if self.use_noise and self.training:
            # denoise additive noise
            # out = [B,O] - [B,I][O,I]^T
            out = out - torch.nn.functional.linear(noise, W, bias=None)
            
            # denoise multiplicative noise
            #out /= noise
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

