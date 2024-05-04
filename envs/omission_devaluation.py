import numpy as np

LEVER = 0 
WITHOLD = 1
class OmissionDevaluationTask:

    def __init__(self, omission: bool = True, num_training_trials: int = 100):
        self._steps = 0
        self.training = True
        self.LEVER = LEVER
        self.WITHOLD = WITHOLD
        self.max_steps_per_episode = 1
        self.num_training_trials = num_training_trials
        self.omission = omission # omission == False -> devaluation

    def reset(self) -> float:
        self._steps = 0
        return np.array(0.1)

    @property
    def num_actions(self) -> int:
        return 2
    
    @property
    def obs_dim(self) -> int:
        return 1
    
    def update(self, ep_idx):
        if ep_idx > self.num_training_trials:
            self.training = False
    
    def obs_to_state(self, obs):
        return obs

    def step(self, action: int) -> float:

        if self.omission:
            if self.training:
                if action == LEVER: reward = np.random.choice([0.0, 1.0], p=[0.5, 0.5])
                elif action == WITHOLD: reward = 0.1 
            else:
                if action == LEVER: reward = 0.0
                elif action == WITHOLD: reward = np.random.choice([0.1, 1.1], p=[0.5, 0.5])
        else: # devaluation 
            if self.training:
                if action == LEVER: reward = np.random.choice([0.0, 1.0], p=[0.5, 0.5])
                elif action == WITHOLD: reward = 0.1 
            else: 
                if action == LEVER: reward = 0.0  
                elif action == WITHOLD: reward = 0.1 

        # increment step counter
        self._steps += 1

        return np.array(0.1), reward, True, {}