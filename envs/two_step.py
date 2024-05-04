import numpy as np
import pdb

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class TwoStepTask:

  def __init__(
    self,
    trials_per_episode=100,
    transition_reversals=False,
    tr_prob=0.9,
    r_prob=0.9,
    tr_switch_prob=0.025,
    r_switch_prob=0.025,
    anneal_r_switch_prob=False,
    reward_scale=1.0,
    show_transition_feature=False,
    **kwargs,
    ):
    
    # four states: fixation state, 1st step state, + the two 2nd step states; obs is one-hot
    self._S = [0, 1, 2, 3]
    self._nS = len(self._S)
    self._start_state = 0
    self._r_probs = np.array([0.0, 0.0, r_prob, 1 - r_prob])
    self._trials_per_episode = trials_per_episode
    self.transition_hist = [] # 0 = common, 1 = uncommon
    self.rewarded_hist = []
    self.stay_hist = []
    self.transition_state_hist = []
    self._prev_action = -1
    self._prev_state = -1
    self._prev_transition_state = -1
    self._trial_steps = 0
    self._transition_reversals = transition_reversals
    self._left_probs = np.array([1 - tr_prob, tr_prob])
    self._right_probs = np.array([tr_prob, 1 - tr_prob])
    self.tr_prob = tr_prob
    self.transition_state = int(self._left_probs[1] == self.tr_prob)
    self._r_switch_prob = r_switch_prob
    self._tr_switch_prob = tr_switch_prob
    self.in_stage0 = False
    self.in_stage1 = False
    self.in_stage2 = False
    self._show_transition_feature = show_transition_feature
    self._anneal_r_switch_prob = anneal_r_switch_prob
    self._reward_scale = reward_scale
    
    
  @property
  def obs_dim(self):
      return self._nS if not self._transition_reversals else self._nS + 1

  @property
  def number_of_states(self):
      return self._nS 
  
  @property
  def max_steps_per_episode(self):
      return 3 * self._trials_per_episode

  @property
  def num_actions(self):
      return 3

  @property
  def goal_states(self):
      return self._goal_hist

  def get_obs(self, s=None):
      base_obs = np.eye(self._nS)[self._state if s is None else s]
      if self._show_transition_feature:
          return np.array(list(base_obs) + [self.transition_state])
          
      return base_obs

  def obs_to_state(self, obs):
    return np.argmax(obs) 

  @property
  def episodes(self):
      return self._episodes

  def reset(self):
    self._trials = 0
    self._state = 0
    self._trial_steps = 0
    self.transition_hist = []
    self.rewarded_hist = []
    self.stay_hist = []
    self.transition_state_hist = []
    return self.get_obs()

  def get_hists(self):
      return self.transition_hist, self.rewarded_hist, self.stay_hist, self.transition_state_hist
  
  def update(self, ep_idx):
    if self._anneal_r_switch_prob:
        if ep_idx < 2000:
            self._r_switch_prob = 0.0 
        else:
            self._r_switch_prob = 0.025 * sigmoid(0.0008 * ep_idx - 1)

  def step(self, action):

    # actions: 
    #   0 = fixate
    #   1 = go left
    #   2 = go right

    done = False
    self.in_stage0, self.in_stage1, self.in_stage2 = False, False, False

    if self._state == 0: # fixation
        self.in_stage0 = True
        if action == 0:
            reward = 0
            new_state = 1 # start trial
            if np.random.random() < self._r_switch_prob and self._prev_state != 0: 
                # switch reward contingencies
                tmp = self._r_probs[-1]
                self._r_probs[-1] = self._r_probs[-2]
                self._r_probs[-2] = tmp
            if np.random.random() < self._tr_switch_prob and self._prev_state != 0 and self._transition_reversals:
                # switch transition contingencies
                tmp = self._left_probs
                self._left_probs = self._right_probs
                self._right_probs = tmp
            transition_state = int(self._left_probs[1] == self.tr_prob)
            self.transition_state_hist.append(int(transition_state == self._prev_transition_state))
            self._prev_transition_state = self.transition_state
            self.transition_state = transition_state
        else:
            reward = -1
            new_state = 0
        

    if self._state == 1: # selection
        self.in_stage1 = True
        if action == 1: # left
            reward = 0
            new_state = np.random.choice([2, 3], p=self._left_probs)
            uncommon_state = [2, 3][np.argmin(self._left_probs)]
            self.transition_hist.append(int(new_state == uncommon_state)) # 1 if uncommon, 0 else
            self.stay_hist.append(int(action == self._prev_action))
            self._prev_action = action
        elif action == 2: # right
            reward = 0
            new_state = np.random.choice([2, 3], p=self._right_probs) # eg [0.8, 0.2]
            uncommon_state = [2, 3][np.argmin(self._right_probs)]
            self.transition_hist.append(int(new_state == uncommon_state)) # 1 if uncommon, 0 else
            self.stay_hist.append(int(action == self._prev_action))
            self._prev_action = action
        else:
            reward = -1
            new_state = 1
    
    info = {}
    if self._state in [2, 3]:
        self.in_stage2 = True
        pr_r_s = self._r_probs[self._state] # get probability of reward for current state
        reward = np.random.choice([0, 1], p=[1 - pr_r_s, pr_r_s])
        self.rewarded_hist.append(reward)
        trial_type = "common" if not self.transition_hist[-1] else "uncommon"
        trial_type += "_rewarded" if reward else "_unrewarded"
        info["trial_type"] = trial_type
        info["stay"] = self.stay_hist[-1]
        new_state = 0

    if self._state in [2, 3] and new_state == 0: # if we're back to the start state 
        self._trials += 1

    self._prev_state = self._state
    self._state = new_state

    if self._trials >= self._trials_per_episode:
        done = True

    return self.get_obs(), reward * self._reward_scale, done, info
