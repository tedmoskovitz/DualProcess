from line_profiler import profile
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np


# https://github.com/senya-ashukha/sparse-vd-pytr/blob/master/svdo-solution.ipynb
class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()
        self.sample_weights()

    def sample_weights(self):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.weight))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)

        if self.training:
            self.sampled_weights = Normal(self.weight, torch.exp(self.log_sigma) + 1e-8).rsample()
        else:
            self.sampled_weights = self.weight * (self.log_alpha < 3).float()

    def reset_log_sigma(self):
        self.log_sigma.data.fill_(-5)

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.bias.data.fill_(0)
        self.reset_log_sigma()

    def forward(self, x):
        return F.linear(x, self.sampled_weights, self.bias)

    def sparsity(self):
        log_alpha = self.log_alpha.detach().cpu().numpy()
        log_alpha[log_alpha > 3] = 0 
        x = (log_alpha > 0).astype(int)
        return 1 - np.sum(x) / x.size # fraction of values set to zero 

    def kl(self):
        # Return KL here -- a scalar
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha)) - k1
        a = - torch.sum(kl)
        return a


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, vdo=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._vdo = vdo
        linear = nn.Linear if not vdo else LinearSVDO

        self.i2h = linear(input_size, 4 * hidden_size)
        self.h2h = linear(hidden_size, 4 * hidden_size)
        self._normi2h = input_size * 4 * hidden_size
        self._normh2h = hidden_size * 4 * hidden_size

    def init_weights(self):
        raise NotImplementedError

    def sample_weights(self):
        if self._vdo:
            self.i2h.sample_weights()
            self.h2h.sample_weights()

    def sparsity(self):
        components = [self.i2h, self.h2h]
        avg_sparsity = sum([l.sparsity() for l in components]) / len(components) 
        return avg_sparsity

    def vd_kl(self):
        components = [self.i2h, self.h2h]
        norms = [self._normi2h, self._normh2h]

        avg_kl = sum([l.kl() / n for (l, n) in zip(components, norms)]) / len(norms)

        return avg_kl 

    @profile
    def forward(self, x, state):
        hx, cx = state
        gates = self.i2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.squeeze().chunk(4, dim=-1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # cy = torch.addcmul(forgetgate * cx, ingate, cellgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class SoftAsymmetryLayer(nn.Module):
    """ learned input gating layer """
    def __init__(self, size):
        super().__init__()
        self.size = size
        weights = torch.Tensor(size)
        self.weights = nn.Parameter(weights)  

        # initialize weights and biases
        nn.init.uniform_(self.weights, -1e-3, 1e-3)

    def forward(self, x):
        w_times_x = x * torch.sigmoid(150 * self.weights) 
        return w_times_x 


class AC_LSTM(nn.Module):

    def __init__(self, input_dim, num_actions, hidden_size=64, vdo=False, asymmetry=False, use_critic=True):
        super().__init__()
        self._vdo = vdo 
        self._use_critic = use_critic
        self.asymmetry = SoftAsymmetryLayer(input_dim) if asymmetry else nn.Identity()
        self.network = LSTM(input_dim, hidden_size, vdo=vdo)
        self.state = (torch.randn(1, 1, hidden_size), torch.randn(1, 1, hidden_size))
        self.hidden_size = hidden_size

        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1) if self._use_critic else None

    def sample_weights(self):
        if self._vdo:
            self.network.sample_weights()
    
    # @profile
    def forward(self, x):
        x = self.asymmetry(x)
        hx, (hx, cx) = self.network(x, self.state)
        self.state = (hx, cx)
        pi = F.softmax(self.actor(hx), dim=-1)
        v = torch.tensor(0.0, device=x.device) if not self.critic else self.critic(hx)
        return pi, v

    def detach_state(self):
        hidden, cell = self.state
        self.state = (hidden.detach(), cell.detach())  
    
    def reset_state(self): 
        self.state = (torch.randn(1, 1, self.hidden_size), torch.randn(1, 1, self.hidden_size))

    def sparsity(self):
        return self.network.sparsity()

    def vd_kl(self):
        return self.network.vd_kl()