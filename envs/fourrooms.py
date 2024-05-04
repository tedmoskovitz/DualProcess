import copy
import matplotlib.pyplot as plt
import numpy as np


TOP_LEFT = 12
BOTTOM_RIGHT = 108

class FourRoomsTask:

    def __init__(
        self,
        start_state=100,
        p_common_goal=0.8,
        max_steps_per_episode=75,
        start_rooms=[0,1,2,3],):
        # -1: wall
        # 0: empty, episode continues
        # other: number indicates reward, episode will terminate
        W = -1
        G = 50
        self._W = W  # wall
        self._G = G  # goal
        self.max_steps_per_episode = max_steps_per_episode

        self._layout = np.array([
            [W, W, W, W, W, W, W, W, W, W, W],
            [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W],
            [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
            [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
            [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
            [W, 0, W, W, W, W, W, 0, W, W, W],
            [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
            [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
            [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
            [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
            [W, W, W, W, W, W, W, W, W, W, W], 
        ])
        self._empty_layout = copy.copy(self._layout)
        self._N = self._layout.shape[0]
        self._idx_layout = np.arange(self._layout.size).reshape(self._layout.shape)
        flat_layout = self._layout.flatten()
        self.wall_idxs = np.stack(np.where(flat_layout == W)).T
        # room layout:
        # 1 2
        # 0 3
        self.room0_idxs = list(self._idx_layout[self._N//2:, :self._N//2].flatten())
        self.room1_idxs = list(self._idx_layout[:self._N//2, :self._N//2].flatten())
        self.room2_idxs = list(self._idx_layout[:self._N//2, self._N//2:].flatten())
        self.room3_idxs = list(self._idx_layout[self._N//2:, self._N//2:].flatten())
        self.room0_idxs = [idx for idx in self.room0_idxs if idx not in self.wall_idxs]
        self.room1_idxs = [idx for idx in self.room1_idxs if idx not in self.wall_idxs]
        self.room2_idxs = [idx for idx in self.room2_idxs if idx not in self.wall_idxs]
        self.room3_idxs = [idx for idx in self.room3_idxs if idx not in self.wall_idxs]
        self._room_idxs = [self.room0_idxs, self.room1_idxs, self.room2_idxs, self.room3_idxs]
        self.start_rooms = start_rooms
        self._possible_reward_states = [] 
        self._possible_start_states = [] 
        for sr in self.start_rooms:
            self._possible_start_states += self._room_idxs[sr]

        self._number_of_states = np.prod(np.shape(self._layout))

        # possible reward states are those where there isn't a wall
        self.r = np.zeros(self._number_of_states)
        self.p_common_goal = p_common_goal
        goal_state = TOP_LEFT if np.random.random() < p_common_goal else BOTTOM_RIGHT
        self._goal_hist = [goal_state] 
        self.goal_state = goal_state
        self.r[goal_state] = G
        rg, cg = self.obs_to_state_coords(goal_state)
        self._layout[rg, cg] = G

        self._random_start = start_state < 0
        self.default_start = start_state
        if self._random_start:
            self._start_state = self.obs_to_state_coords(np.random.choice(self._possible_start_states))
        else:
            self._start_state = self.obs_to_state_coords(start_state)
        self._episodes = 0
        self._state = self._start_state
        self._start_obs = self.get_obs()
        self._number_of_states = np.prod(np.shape(self._layout))
        self._steps = 0

    @property
    def obs_dim(self):
        return 4
    
    @property
    def num_actions(self):
        return 4
    
    def update(self, ep_idx):
        return

    def set_start_rooms(self, rooms):
        self._possible_start_states = [] 
        self.start_rooms = rooms
        for room in rooms:
            self._possible_start_states += self._room_idxs[room]

    def set_goal(self, coords):
        rg, cg = self.obs_to_state_coords(self.goal_state)
        self._layout[rg, cg] = 0
        rg, cg = coords 
        sg = rg*self._layout.shape[1] + cg
        self.goal_state = sg
        self.r[sg] = self._G
        self._layout[rg, cg] = self._G

    @property
    def number_of_states(self):
        return self._number_of_states

    @property
    def goal_states(self):
        return self._goal_hist

    def get_obs(self, s=None):
        r, c = self._state if s is None else s
        # gr, gc = self.obs_to_state_coords(self.goal_state)
        goal = [1, 0] if self.goal_state == TOP_LEFT else [0, 1]
        # idx = y*self._layout.shape[1] + x
        return np.array([r, c] + goal)

    def obs_to_state(self, obs):
        # x = obs % self._layout.shape[1]
        # y = obs // self._layout.shape[1]
        r, c, g_common, _ = obs
        g_idx = TOP_LEFT if g_common == 1 else BOTTOM_RIGHT
        gc = g_idx % self._layout.shape[1]
        gr = g_idx // self._layout.shape[1]
        s = np.copy(self._layout)
        s[r, c] = 4
        s[gr, gc] = self._G
        return s

    def obs_to_state_coords(self, obs):
        x = obs % self._layout.shape[1]
        y = obs // self._layout.shape[1]
        return (y, x)

    @property
    def episodes(self):
        return self._episodes

    def reset(self):
        if self._random_start:
            self._start_state = self.obs_to_state_coords(np.random.choice(self._possible_start_states))

        self._state = self._start_state
        self._episodes = 0
        self.r = np.zeros(self._number_of_states)
        # reset old goal loc to 0
        rg, cg = self.obs_to_state_coords(self.goal_state)
        self._layout[rg, cg] = 0
        goal_state = TOP_LEFT if np.random.random() < self.p_common_goal else BOTTOM_RIGHT
        self._goal_hist.append(goal_state)
        self.goal_state = goal_state
        self.r[goal_state] = self._G
        rg, cg = self.obs_to_state_coords(goal_state)
        self._layout[rg, cg] = self._G
        self._steps = 0
        return self.get_obs()

    def step(self, action):
        done = False
        y, x = self._state
        r2d = np.reshape(self.r, self._layout.shape)
        
        if action == 0:  # up
            new_state = (y - 1, x)
        elif action == 1:  # right
            new_state = (y, x + 1)
        elif action == 2:  # down
            new_state = (y + 1, x)
        elif action == 3:  # left
            new_state = (y, x - 1)
        else:
            raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

        new_y, new_x = new_state
        reward = self._layout[new_y, new_x]
        if self._layout[new_y, new_x] == self._W:  # wall
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0 and r2d[new_y, new_x] == 0:  # empty cell
            pass
        else:  # a goal
            self._episodes += 1
            reward = r2d[new_y, new_x]
            done = True

        self._state = new_state
        self._steps += 1

        if self._steps >= self.max_steps_per_episode:
            done = True
        return self.get_obs(), reward, done, {}

    def plot_grid(
        self,
        traj=None, M=None, ax=None, goals=None,
        cbar=False, traj_color="C2", title='FourRooms', show_idxs=False,
        cmap='viridis', show_goal=True, vmin=None, vmax=None, show_start=True
        ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        layout = copy.copy(self._layout).astype(float)
        if traj is None and goals is not None:
            for g in goals:
                gx, gy = self.obs_to_state_coords(g)
                layout[gx, gy] = np.nan
        cmap = plt.cm.pink 
        cmap.set_bad(color="C6")
        layout[layout == 0] = 50.0
        layout[5, 7] = 50.0
        ax.imshow(layout, interpolation="nearest", cmap='pink')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=30)
        startx, starty = self._start_state
        goalx, goaly = self.obs_to_state_coords(np.argmax(self.r))
        if show_start:
            ax.text(starty, startx, r"$\mathbf{s_0}$", ha='center', va='center', fontsize=16)
        if traj is None and show_goal and goals is None:
            ax.text(goaly, goalx, r"$\mathbf{s_g}$", ha='center', va='center', fontsize=16)
        
        if show_idxs:
            for i in range(self._layout.shape[0]):
                for j in range(self._layout.shape[1]):
                    ax.text(j, i, f"{self.get_obs(np.array([i, j]))[0]}", ha='center', va='center')
        
        h, w = self._layout.shape
        for y in range(h-1):
            ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
        for x in range(w-1):
            ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

        if traj is not None:
            # plot trajectory, list of [(y0, x0), (y1, x1), ...]
            if goals is None:
                traj = np.vstack(traj)
                ax.plot(traj[:, 1], traj[:, 0], c=traj_color, lw=3)
            else:
                # draw goals
                for i,g in enumerate(goals):
                    if g != np.argmax(self.r):
                        y, x = self.obs_to_state_coords(g)
                        ax.text(x, y, r"$\mathbf{s_g}$", ha='center', va='center', fontsize=16, color=f'C{i}')
                # draw trajectories
                traj = np.vstack(traj)
                ax.plot(traj[:, 1], traj[:, 0], c=traj_color, lw=3, ls='-')


if __name__ == "__main__":

    env = FourRoomsTask(start_state=-1, p_common_goal=0.5)
    t = 0
    obs = env.reset()
    done = False
    print(env.obs_to_state(obs))
    action = int(input("Enter an action:"))
    if action not in list(range(4)):
        raise ValueError(f"Invalid action: {action} is not 0, 1, 2, or 3.")

    while not done:
        obs, reward, done, _ = env.step(action)
        print(env.obs_to_state(obs), reward, done)
        action = int(input("Enter an action:"))
        t += 1

