import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

# Taken mostly from https://github.com/EkdeepSLubana/BeyondBatchNorm

### BatchNorm (Ioffe and Szegedy, 2015) ###
# (based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d)
class BN_self(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('moving_mean', torch.ones(shape))
        self.register_buffer('moving_var', torch.ones(shape))
        self.reset_parameters()

    def reset_parameters(self):
        self.moving_var.fill_(1)

    def forward(self, X):
        if self.training:
            var, mean = torch.var_mean(X, dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.moving_mean.mul_(self.momentum)
            self.moving_mean.add_((1 - self.momentum) * mean)
            self.moving_var.mul_(self.momentum)
            self.moving_var.add_((1 - self.momentum) * var)
        else:
            var = self.moving_var
            mean = self.moving_mean

        X = (X - mean) * torch.rsqrt(var+self.eps)
        return X * self.gamma + self.beta


############### Prober layers for tracking statistics ###############
### Conv_prober (i.e., probes activations and gradients from convolutional layers) ###
class Conv_prober(nn.Module):
    def __init__(self):
        super(Conv_prober, self).__init__()
        self.std_list = []
        # Grads
        self.grads_norms = []

        class sim_grads(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                self.std_list.append(input.std(dim=[0,2,3]).mean().item())
                return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                M = grad_output.view(grad_output.shape[0], -1)
                # Gradient Norms
                self.grads_norms.append(M.norm().item())
                M = (M / (torch.linalg.norm(M, dim=[1], keepdim=True)+1e-10))
                M = torch.matmul(M, M.T)
                return grad_output.clone()

        self.cal_prop = sim_grads.apply

    def forward(self, input):
        if not torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input)


### Activs_prober (i.e., probes activations) ###
class Activs_prober(nn.Module):
    def __init__(self):
        super(Activs_prober, self).__init__()
        # Activs
        self.activs_norms = []
        self.activs_corr = []
        self.activs_ranks = []

        class sim_activs(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                M = input.clone()
                # Activation Variance
                avar = M.var(dim=[0,2,3], keepdim=True)
                self.activs_norms.append(avar.mean().item())
                anorms = torch.linalg.norm(M, dim=[1,2,3], keepdim=True)
                # self.activs_norms.append(anorms.mean().item())
                M = (M / anorms).reshape(M.shape[0], -1)
                M = torch.matmul(M, M.T)
                # Activation Correlations
                self.activs_corr.append(((M.sum(dim=1) - 1) / (M.shape[0]-1)).mean().item())
                # Activation Ranks (calculates stable rank)
                tr = torch.diag(M).sum()
                opnom = torch.linalg.norm(M, ord=2)
                self.activs_ranks.append((tr / opnom).item())
                return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.clone()

        self.cal_prop = sim_activs.apply

    def forward(self, input):
        if not torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input)

### VGG
class VGG(nn.Module):
    def __init__(self, cfg, n_classes=100, p_grouping=1.0, probe=True, group_list=None):
        super(VGG, self).__init__()
        self.probe = probe
        self.p_grouping = p_grouping
        self.group_list = group_list

        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-1], n_classes)

        if self.probe:
            self.params_list = []
            self.grads_list = []
            for _ in cfg:
                self.params_list.append([])
                self.grads_list.append([])

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        s = nn.Sequential()
        for gid, x in enumerate(cfg):
            if type(x) == tuple:
                n_groups = (int(self.p_grouping) if (self.p_grouping>1) else int(np.ceil(x[0] * self.p_grouping))) if self.group_list==None else self.group_list[gid]
                s.add_module("conv{}".format(gid), nn.Conv2d(in_channels, x[0], kernel_size=3, padding=1, stride=2))
                s.add_module("conv_prober{}".format(gid), Conv_prober() if self.probe else nn.Identity())
                s.add_module("bn{}".format(gid), BN_self(x[0]))
                s.add_module("activs{}".format(gid), nn.ReLU(inplace=True))
                s.add_module("activ_prober{}".format(gid), Activs_prober() if self.probe else nn.Identity())

                # layers += [nn.Conv2d(in_channels, x[0], kernel_size=3, padding=1, stride=2),
                #            Conv_prober() if self.probe else nn.Identity()]
                # layers += [BN_self(x[0]), nn.ReLU(inplace=True)]
                # layers += [Activs_prober() if self.probe else nn.Identity()]
                in_channels = x[0]
            else:
                n_groups = (int(self.p_grouping) if (self.p_grouping>1) else int(np.ceil(x * self.p_grouping))) if self.group_list==None else self.group_list[gid]
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           Conv_prober() if self.probe else nn.Identity()]
                layers += [BN_self(x), nn.ReLU(inplace=True)]
                layers += [Activs_prober() if self.probe else nn.Identity()]
                in_channels = x
        # return nn.Sequential(*layers)
        return s

    def forward(self, x):
        out = self.features(x)
        out = out.mean(dim=(2,3))
        out = self.classifier(out)
        return out
