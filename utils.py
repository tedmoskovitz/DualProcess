from dataclasses import dataclass
import numpy as np
import random
import torch

@dataclass
class AgentOutput:
    log_prob: torch.Tensor
    pi: torch.Tensor
    value: torch.Tensor
    pi0: torch.Tensor
    a_vd_kl: torch.Tensor = None
    s_vd_kl: torch.Tensor = None

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_beta_schedule(beta_max, number_of_episodes, vdo_start=0.5):
    if beta_max == 0:
       return np.zeros(number_of_episodes+1)
    n_start = int(vdo_start * number_of_episodes) 
    beta_schedule = np.zeros(number_of_episodes+1)
    n_warm = number_of_episodes // 100
    beta_schedule[n_start: n_start + n_warm] = np.linspace(0, beta_max, num=n_warm)
    beta_schedule[n_start + n_warm:] = beta_max
    return beta_schedule


def KL(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
  """compute KL(p||q)"""
  return (p * torch.log(p + eps) - p * torch.log(q + eps)).sum(dim=-1)#.mean()

def entropy(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
  """compute entropy of a distribution"""
  return -(p * torch.log(p + eps)).sum(dim=-1)

