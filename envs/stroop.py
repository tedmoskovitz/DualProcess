import pdb
from termcolor import colored
import numpy as np



COLOR_NAMING = 0
COLOR_NAMING2 = -1
WORD_NAMING = 1

class StroopTask:
    
    def __init__(self, max_trials=100, flatten=False, training=True, uniform=False, verbose=False):
        """
        **currently configured s.t. only get reward when match the "color" feature
        could add a cue feature that says whether to match color or word 

        two colors: red, blue
        text can be any of the above

        2d state space [word feature, color feature], each feature is {0, 1, 2, 3} 
            -> this makes 4 x 4 = 16 possible states
        4d action space, corresponding to each color/text value 

        get +1.0 reward when action == color feature
        get -0.1 reward otherwise 
        """
        super().__init__()
        
        self._colors = ["blue", "red"]
        self._n = len(self._colors)
        self._flatten = flatten
        self.training = training
        self.verbose = verbose

        # item-specific stroop
        B, R = -1, 1
        self.BLUE, self.RED = B, R
        self.WR, self.CN = WORD_NAMING, COLOR_NAMING2
        self._stims = [
            # color, word, task
            [B, B, WORD_NAMING],
            [R, R, WORD_NAMING],
            [B, R, WORD_NAMING],
            [R, B, WORD_NAMING],
            [B, B, COLOR_NAMING2],
            [R, R, COLOR_NAMING2],
            [B, R, COLOR_NAMING2],
            [R, B, COLOR_NAMING2],
        ]
        self._K = len(self._stims)
        self._probs = [
            0.2, 0.2, 0.2, 0.2,
            0.05, 0.05, 0.05, 0.05
        ]
        if uniform:
            self._probs = self._K * [1 / self._K]
        self.error_rates = np.zeros(self._K)
        self.stim_counts = np.ones(self._K)
        self.error_hist = []
        
        self._max_trials = max_trials
        self.max_steps_per_episode = 1
        self._trials = 0
        state_idx = np.random.choice(self._K, p=self._probs) # add task id 
        self._state_idx = state_idx
        self.stim_counts[state_idx] += 1
        self._state = np.array(self._stims[state_idx])

    @property
    def number_of_states(self):
        return int(self._n ** 2) * 2
    
    @property
    def obs_dim(self):
        return 3

    @property
    def num_actions(self):
        return 2

    def get_error_rates(self):
        result = []
        for i in range(self._K):
            if self.stim_counts[i] > 1:
                result.append(self.error_rates[i] / (self.stim_counts[i] - 1))
            else:
                result.append(1)
        return np.array(result)

    def state_to_obs(self, state):
        # state space is a n (words) x n (colors) x 2 (tasks) tensor -> obs is the flattened index
        word, color = state
        return word * self._n + color

    def get_obs(self):
        obs = self.state_to_obs(self._state) if self._flatten else self._state
        return obs
    
    def obs_to_state(self, obs):
        return obs
    
    def update(self, ep_idx):
        return
        
    def step(self, action):
        
        # get reward if agent picks the right option
        self.error_hist.append(np.nan_to_num(self.error_rates / self.stim_counts, nan=1.0, posinf=1.0))
        reward = 0
        color, word, task_id = self._state
        action = -1 if action == 0 else action
        if task_id == COLOR_NAMING2:
            if action == color: reward = 1
        elif task_id == WORD_NAMING:
            if action == word: reward = 1

        if self.verbose:
            print (f"color naming? {task_id == COLOR_NAMING2}: color = {color}, word = {word}, action = {action}, reward = {reward}")

        self.error_rates[self._state_idx] += float(reward != 1) 
        
        # observation at each step is a random word-color combo
        state_idx = np.random.choice(self._K, p=self._probs) # add task id 
        self.stim_counts[state_idx] += 1
        self._state_idx = state_idx
        self._state = np.array(self._stims[state_idx])
        
        # if in training mode, remove the irrelevant feature of the task (set it to 0)
        if self.training:
            color, word, task_id = self._state
            if task_id == COLOR_NAMING2: self._state[WORD_NAMING] = 0 
            elif task_id == WORD_NAMING: self._state[COLOR_NAMING] = 0

        self._trials += 1
        done = True
        
        obs = self.state_to_obs(self._state) if self._flatten else self._state
        return obs, reward, done, {}
    
    def reset(self):
        
        self._trials = 0
        
        # observation is a random word-color combo
        state_idx = np.random.choice(self._K, p=self._probs) # add task id 
        self._state_idx = state_idx
        self.stim_counts[state_idx] += 1
        self._state = np.array(self._stims[state_idx])
        # if in training mode, remove the irrelevant feature of the task (set it to 0)
        if self.training:
            color, word, task_id = self._state
            if task_id == COLOR_NAMING2: self._state[WORD_NAMING] = 0 
            elif task_id == WORD_NAMING: self._state[COLOR_NAMING] = 0
        
        obs = self.state_to_obs(self._state) if self._flatten else self._state
        return obs
    
    def render(self, s):
        # word, color
        s = s if s is not None else self._state
        color, word, task = s
        task_str = "WR" if task == WORD_NAMING else "CN"
        print(f"task = {task_str}")
        word = "red" if word == self.RED else "blue"
        color = "red" if color == self.RED else "blue"
        print(colored(word, color))



