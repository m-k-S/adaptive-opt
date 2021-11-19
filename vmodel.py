import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

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
        # self.activs_ranks = []

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        class sim_activs(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                M = input.clone()
                # Activation Variance
                # avar = M.var(dim=[0,2,3], keepdim=True)
                # self.activs_norms.append(avar.mean().item())
                anorms = torch.linalg.norm(M, dim=[1,2,3], keepdim=True)
                self.activs_norms.append(anorms.mean().item())
                M = (M / anorms).reshape(M.shape[0], -1)
                M = torch.matmul(M, M.T)
                # Activation Correlations
                self.activs_corr.append(((M.sum(dim=1) - 1) / (M.shape[0]-1)).mean().item())
                # Activation Ranks (calculates stable rank)
                # tr = torch.diag(M).sum()
                # opnom = torch.linalg.norm(M + 1e-8 * M.mean() * torch.rand(M.shape).to(self.device), ord=2)
                # self.activs_ranks.append((tr / opnom).item())
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

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        self.params_list = []
        self.grads_list = []
        for _ in features:
            self.params_list.append([])
            self.grads_list.append([])

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1), Conv_prober()]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True), Activs_prober()]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))
