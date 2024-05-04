import pdb
import torch
from typing import List
from utils import AgentOutput

class TrajectoryReplay:

    def __init__(self, capacity, trajectory_len, num_actions):
        self.capacity = capacity
        self.num_actions = num_actions
        self.trajectory_len = trajectory_len
        self._ptr = 0
        self.log_probs = torch.zeros((capacity, trajectory_len))
        self.pis = torch.zeros((capacity, trajectory_len, num_actions))
        self.values = torch.zeros((capacity, trajectory_len))
        self.pi0s = torch.zeros((capacity, trajectory_len, num_actions))
        self.rewards = torch.zeros((capacity, trajectory_len))
        self.dones = torch.zeros((capacity, trajectory_len), dtype=torch.int64)

        # self.log_probs = torch.zeros((capacity, trajectory_len))
        # self.pis = torch.zeros((capacity, trajectory_len, num_actions))
        # self.values = torch.zeros((capacity, trajectory_len))
        # self.pi0s = torch.zeros((capacity, trajectory_len, num_actions))
        # self.rewards = torch.zeros((capacity, trajectory_len))
        # self.dones = torch.zeros((capacity, trajectory_len), dtype=torch.int64)

    def __call__(self, t: int):
        return (self.log_probs[:, t],
                self.pis[:, t],
                self.values[:, t],
                self.pi0s[:, t],
                self.rewards[:, t],
                self.dones[:, t])

    def add(self, trajectory_outs: List[AgentOutput], trajectory_rewards: List[float]):
        for t, (output, rew) in enumerate(zip(trajectory_outs, trajectory_rewards)):
            # pdb.set_trace()
            self.log_probs[self._ptr, t] = output.log_prob
            self.pis[self._ptr, t] = output.pi
            self.values[self._ptr, t] = output.value
            self.pi0s[self._ptr, t] = output.pi0
            self.rewards[self._ptr, t] = rew
        
        self.dones[self._ptr, t:] = 1

        # T = len(trajectory_outs)
        # self.log_probs[self._ptr, :T] = torch.tensor([output.log_prob for output in trajectory_outs])
        # self.log_probs[self._ptr, T:] = 0.0
        # self.pis[self._ptr, :T] = torch.stack([output.pi for output in trajectory_outs])
        # self.pis[self._ptr, T:] = 0.0
        # self.values[self._ptr, :T] = torch.tensor([output.value for output in trajectory_outs])
        # self.values[self._ptr, T:] = 0.0
        # self.pi0s[self._ptr, :T] = torch.stack([output.pi0 for output in trajectory_outs])
        # self.pi0s[self._ptr, T:] = 0.0
        # self.rewards[self._ptr, :T] = torch.tensor(trajectory_rewards)
        # self.rewards[self._ptr, T:] = 0.0
        # self.dones[self._ptr, T-1:].fill_(1)
        # self.dones

        # pdb.set_trace()
        self._ptr += 1
        if self._ptr == self.capacity:
            self._ptr = 0
    
    def clear(self):
        self._ptr = 0
        self.log_probs = torch.zeros_like(self.log_probs)
        self.pis = torch.zeros_like(self.pis)
        self.values = torch.zeros_like(self.values)
        self.pi0s = torch.zeros_like(self.pi0s)
        self.rewards = torch.zeros_like(self.rewards)
        self.dones = torch.zeros_like(self.dones)

        

