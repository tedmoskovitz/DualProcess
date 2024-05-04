import numpy as np

class DriftingBanditTask:

    def __init__(self, drift_std=0.15):
        self.drift_std = drift_std
        self.p_left = np.random.uniform()
        self.p_right = np.random.uniform()
        self._trials = 0
        self.p_right_hist = [self.p_right]
        self.p_left_hist = [self.p_left]
        self.max_steps_per_episode = 1

    def reset(self):
        self.p_left = np.random.uniform()
        self.p_right = np.random.uniform()
        self.p_left_hist = []
        self.p_right_hist = []
        self._trials = 0
        return np.array(0.1)

    @property
    def num_actions(self):
        return 2
    
    @property
    def obs_dim(self):
        return 1
    
    def update(self, ep_idx):
        return
    
    def obs_to_state(self, obs):
        return obs

    def step(self, action: int):

        # if choose left (0) port
        if action == 0:
            reward = np.random.choice([-1, 1], p=[1 - self.p_left, self.p_left])
        # if choose right (1) port
        elif action == 1:
            reward = np.random.choice([-1, 1], p=[1 - self.p_right, self.p_right])
        else: raise ValueError("invalid action")

        # update arm probs according to a random walk 
        self.p_left = np.random.normal(loc=self.p_left, scale=self.drift_std)
        self.p_right = np.random.normal(loc=self.p_right, scale=self.drift_std)
        # clip so within [0, 1]
        self.p_left = np.clip(self.p_left, 0, 1)
        self.p_right = np.clip(self.p_right, 0, 1)
        self.p_left_hist.append(self.p_left)
        self.p_right_hist.append(self.p_right)

        # increment step counter
        self._trials += 1

        return np.array(0.1), reward, True, {}

