import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt()+self.eps
        x/=norm.expand_as(x)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
