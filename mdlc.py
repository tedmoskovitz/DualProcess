from collections import namedtuple
import copy
from line_profiler import profile
import numpy as np
import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from networks import AC_LSTM
from replay import TrajectoryReplay
from utils import KL, AgentOutput
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MDLCAgent(nn.Module):
    
    def __init__(self, config: dict) -> None:
        super().__init__()
        obs_dim = config['obs_dim']
        num_actions = config['num_actions']
        hidden_size = config.get('hidden_size', 32)
        alpha = config.get('alpha', 0.05)
        lr = config.get('lr', 4e-3)
        use_a_tm1 = config.get('use_a_tm1', True)
        gamma = config.get('gamma', 0.99)
        value_loss_wt = config.get('value_loss_wt', 0.05)
        vdo = config.get('vdo', False)
        asymmetry = config.get('asymmetry', True)
        default_critic = config.get('default_critic', False)
        rt_budget = config.get('rt_budget', 1)
        rt_threshold = config.get('rt_threshold', 0.5)
        self.batch_size = config.get('batch_size', 1)
        self.max_steps = config.get('max_steps_per_episode', 50)
        self.control_buffer = TrajectoryReplay(self.batch_size, self.max_steps, num_actions)

        obs_dim = obs_dim
        obs_dim += 1  # prev reward
        if use_a_tm1:
            obs_dim += num_actions

        # define associative network
        self.control = AC_LSTM(
            obs_dim,
            num_actions,
            hidden_size=hidden_size,
            vdo=False,
            use_critic=True,
            asymmetry=False
        )

        self.default = AC_LSTM(
            obs_dim,
            num_actions,
            hidden_size=hidden_size,
            vdo=vdo,
            use_critic=default_critic,
            asymmetry=asymmetry
        )

        self._vdo = vdo
        self.device = device
        self.rt_budget = rt_budget
        self.rt_threshold = rt_threshold

        self._lr, self.hidden_size, self.obs_dim = lr, hidden_size, obs_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.num_actions = num_actions

        self._use_a_tm1 = use_a_tm1
        self._a_tm1 = torch.tensor([np.random.choice(num_actions)])
        self._value_loss_wt = value_loss_wt

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # action and reward buffers
        self.gamma = gamma  # 1.0
        self.loss_hist, self.kl_hist, self.v_hist, self.w_hist = [], [], [], []
        if hasattr(self.default.asymmetry, "weights"):
            self.w_hist.append(copy.copy(self.default.asymmetry.weights.detach().numpy()))

    def detach_state(self):
        # detach gradient flow
        self.control.reset_state() 
        self.default.reset_state()
        
    @profile
    def forward(
        self,
        xG: torch.tensor,
        a_tm1: torch.tensor,
        r_tm1: torch.tensor,
        verbose: bool = False
        ):
        a_tm1 = F.one_hot(a_tm1.to(torch.int64), num_classes=self.num_actions).float().squeeze()
        if len(xG.shape) > 1: 
            assert xG.shape[0] == a_tm1.shape[0] and xG.shape[0] == r_tm1.shape[0], "inputs batched incorrectly"

        # input is current state, prev action, and prev reward
        x = torch.hstack([xG, a_tm1, r_tm1])
        # control pathway 
        pi, v = self.control(x)
        # default pathway
        pi0, _ = self.default(x)

        return pi.squeeze(), v.squeeze(), pi0.squeeze()

    def sample_weights(self):
        if self._vdo:
            self.default.sample_weights()

    def reset_control(self):
        self.control = self.ACNet(
            self.obs_dim,
            self.num_actions,
            hidden_size=self.hidden_size,
            vdo=False, 
            use_critic=True,
            asymmetry=False
        )
        self.control_optimizer = torch.optim.Adam(self.control.parameters(), lr=self._lr)

    # @profile
    def get_action(self, state, a_tm1, r_tm1, greedy=False, use_default=False, get_policies=False):
        state = torch.from_numpy(state).float()
        # get action distribution
        
        for t in range(self.rt_budget):
            pi, v, pi0 = self.forward(state, torch.tensor(a_tm1).to(torch.int64), torch.tensor(r_tm1).float()) 

            # convert to a Categorical distribution object
            policy = Categorical(pi) if not use_default else Categorical(pi0)
            # sample an action from behavioral distribution
            action = policy.sample() if not greedy else torch.argmax(pi)

            if policy.entropy() <= self.rt_threshold:
                break

        if get_policies:
            return action.item(), policy.log_prob(action), pi, v, pi0, t
        
        return action.item()

    def update(self, beta=1.0, **kwargs):
        """
        Update control and default policies 
        """
        returns_BT = torch.zeros(self.batch_size, self.max_steps)
        rewards_BT = self.control_buffer.rewards
        dones_BT = self.control_buffer.dones
        for i, t in enumerate(reversed(range(self.max_steps))):
            returns_BT[:, i] = rewards_BT[:, t] + (1 - dones_BT[:, t]) * self.gamma * returns_BT[:, i-1] # should be zeros after ep end

        # control policy
        values_BT = self.control_buffer.values
        log_probs_BT = self.control_buffer.log_probs
        pi0s_BTA = self.control_buffer.pi0s
        pis_BTA = self.control_buffer.pis
        advantages_BT = (returns_BT - torch.flip(values_BT, dims=[1])).detach()
        policy_losses_BT = -torch.flip(log_probs_BT, dims=[1]) * advantages_BT
        policy_kls_BT = KL(pi0s_BTA.detach(), pis_BTA)
        policy_losses_BT = policy_losses_BT + self.alpha * torch.flip(policy_kls_BT, dims=[1])
        value_losses_BT = self._value_loss_wt * F.smooth_l1_loss(values_BT, returns_BT.detach(), reduction='none')

        # default policy
        default_policy_kl_BT = KL(pis_BTA.detach(), pi0s_BTA)
        default_vdo_kl = torch.tensor(0.0) if not self._vdo else self.default.vd_kl()

        # control update ===================================
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        policy_loss = policy_losses_BT.mean() #sum(dim=1).mean()
        value_loss = value_losses_BT.mean() #sum(dim=1).mean()
        default_policy_kl = default_policy_kl_BT.mean() #sum(dim=1).
        default_loss = default_policy_kl + beta * default_vdo_kl
        loss = policy_loss + value_loss + default_loss
        # perform backprop
        loss.backward()
        

        try:
            control_grad_norms = [p.grad.norm().item() for p in self.control.parameters()]
            control_grad_norm = sum(control_grad_norms) / len(control_grad_norms)
        except:
            control_grad_norm = 0.0
        try:
            default_grad_norms = [p.grad.norm().item() for p in self.default.parameters()]
            default_grad_norm = sum(default_grad_norms) / len(default_grad_norms)
        except:
            default_grad_norm = 0.0

        self.optimizer.step()

        self.clear_buffers()

        metrics = dict(
            default_loss=default_loss.item(),
            kl_pi2pi0=default_policy_kl_BT.mean().item(),
            kl_pi02pi=policy_kls_BT.mean().item(),
            vdo_kl=default_vdo_kl.item(),
            control_policy_loss=policy_loss.item(),
            control_value_loss=value_loss.item(),
            control_grad_norm=control_grad_norm,
            default_grad_norm=default_grad_norm
        )
        return metrics
    
    def clear_buffers(self):
        self.detach_state()
        self.control_buffer.clear()

