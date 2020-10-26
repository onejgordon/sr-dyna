import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.animation as manimation
import random


class SimpleGridWorld():
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    ACTION_LABELS = ["L", "R", "U", "D"]

    def __init__(self, w=5, h=5, max_reward_locs=1, world='worlds/latent_learning.txt'):
        self.max_reward_locs = max_reward_locs
        self.load_world(world)
        self.map = self.get_map()
        self.reward_locs = {}  # loc -> {reward, index}
        self.terminal_state = self.n_states() - 1  # Last index

        self.actions = [self.LEFT, self.RIGHT, self.UP, self.DOWN]

    def load_world(self, fn):
        self.wall_coords = []
        with open(fn, 'r') as f:
            y = 0
            for line in list(f.readlines())[::-1]:
                self.w = len(line) - 1
                for x, cell in enumerate(line):
                    if cell == '1':
                        self.wall_coords.append((x, y))
                y += 1
        self.h = y
        print("Loaded %dx%d world with %d states" % (self.w, self.h, self.n_states()))

    def add_reward(self, loc, amount=10):
        new_idx = len(self.reward_locs)
        s = self.count_cells() + new_idx
        self.reward_locs[loc] = {'reward': amount, 'index': new_idx, 's': s}

    def wall_at(self, loc):
        return self.map[loc[1], loc[0]] == 1

    def get_map(self):
        m = self.get_space()
        if self.wall_coords:
            for mc in self.wall_coords:
                m[mc[1], mc[0]] = 1
        return m

    def state_at_loc(self, loc, ignore_reward=False):
        if not ignore_reward and self.reward_loc(loc):
            ro = self.reward_locs.get(loc)
            s = ro.get('s')
        else:
            x, y = loc
            s = self.w * y + x
        return s

    def loc_at_state(self, s):
        x = int(s % self.w)
        y = int(s / self.w)
        return (x, y)

    def reward_state(self, s):
        # TODO: Efficiency
        cells = self.count_cells()
        is_reward_state = s in range(cells, cells+self.max_reward_locs)
        if is_reward_state:
            for loc, ro in self.reward_locs.items():
                if ro.get('s') == s:
                    return ro.get('reward')
        return 0

    def reward_loc(self, loc):
        return loc in self.reward_locs.keys()

    def n_states(self):
        return self.count_cells() + self.max_reward_locs + 1

    def count_cells(self):
        return self.w * self.h

    def available_actions(self, s):
        if s >= self.count_cells():
            available = [0]
        else:
            available = self.actions
        return available

    def random_action(self, s):
        aa = self.available_actions(s)
        return random.choice(aa)

    def successor(self, s, a):
        delta = {
            self.LEFT: (-1, 0),
            self.RIGHT: (1, 0),
            self.UP: (0, 1),
            self.DOWN: (0, -1),
        }[a]
        done = False
        reward = 0
        loc = self.loc_at_state(s)
        reward = self.reward_state(s)
        if reward:
            next_s = self.terminal_state
            done = True
        else:
            next_loc = (loc[0] + delta[0], loc[1] + delta[1])
            if not self.valid_loc(next_loc):
                next_s = s  # Revert
            else:
                next_s = self.state_at_loc(next_loc)
        return (next_s, reward, done)

    def valid_loc(self, loc):
        x, y = loc
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return False
        if self.wall_at(loc):
            return False
        return True

    def valid_state(self, s):
        return self.valid_loc(self.loc_at_state(s))

    def get_space(self):
        m = np.zeros((self.h, self.w))
        return m

    def dim(self):
        return self.get_space().size

    def render(self, a, ax=None, last_agent_state=True, last_k_steps=0):
        if ax is None:
            fig, ax = plt.subplots()
        m = self.get_map()
        ax.imshow(m, origin='bottom')
        if a.last_state is not None:
            last_loc = self.loc_at_state(a.last_state)
            render_loc = last_loc if last_agent_state else self.loc_at_state(a.state)
            agent = patches.Circle(render_loc, radius=0.5, fill=True, color='white')
            ax.add_patch(agent)
        for loc in self.reward_locs.keys():
            reward = patches.Circle(loc, radius=0.5, fill=True, color='green')
            ax.add_patch(reward)
        if last_k_steps:
            for t, s, action, s_n, _ in a.replay_buffer[-last_k_steps:]:
                age = a.t - t
                loc = self.loc_at_state(s)
                a_step = patches.Circle(loc, radius=0.2, fill=True, color='white', alpha=1. - (age/last_k_steps))
                ax.add_patch(a_step)
                # print(t, s, loc, action, s_n)
        ax.set_title('Map')
        ax.set_axis_off()


class SRDyna():

    def __init__(self, id, env, loc=(0, 0), **params):
        self.id = id
        self.t = 0
        self.ep_t = 0
        self.start_loc = loc
        self.env = env
        self.n_actions = len(env.actions)
        self.action = None
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        n_state_actions = self.env.count_cells() * self.n_actions \
            + env.max_reward_locs + 1  # 1 for terminal

        # Params
        self.alpha_sr = 0.3
        self.alpha_w = params.get('alpha_w', 0.3)
        self.gamma = 0.95
        self.eps = 0.1
        self.post_step_replays = params.get('post_step_replays', 10)
        self.exp_lambda = params.get('exp_lambda', 1./5)

        # Model
        self.H = params.get('H', np.eye(n_state_actions, n_state_actions))  # H(s, a) -> Expected discounted state-action occupancy
        self.H[-1, -1] = 0  # Terminal state zero'd
        self.W = np.zeros(n_state_actions)  # Value function weights (w(sa) -> E_a[R(s, a)])

        # Memory
        self.replay_buffer = np.array([], dtype=(int, 5))  # Append to end

        # self.Particle = recordclass('Particle', 'loc last_loc energy')  # loc as Node ID

        self.state = self.initial_state()

    def initial_state(self):
        return self.env.state_at_loc(self.start_loc)  # (x, y)

    def terminate_episode(self, reset_state=None):
        if reset_state is None:
            reset_state = self.initial_state()
        self.state = reset_state
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.action = self.eps_greedy_policy(self.state)
        self.ep_t = 0

    def n_states(self):
        return self.env.count_cells()

    def n_viz_sas(self):
        return self.env.count_cells() * self.n_actions

    def state_action_index(self, s, a):
        """
        Index into S x A size array for H
        """
        cells = self.env.count_cells()
        if s < cells:
            return s * self.n_actions + a
        else:
            return cells * self.n_actions + (s - cells)

    def v_pi(self):
        """
        From paper: 'For SR-Dyna, which worked with action values rather than state values,
        the state value function was computed as the max action value available
        in that state.'
        """
        vs = np.matmul(self.H, self.W.reshape(-1, 1))  # 40x x 1
        vs = self.corrected_value_map(vs)
        vs = vs[:self.n_viz_sas()].reshape(-1, 4)
        values = vs.max(axis=1)
        return values

    def q_pi(self, s, a):
        sa_index = self.state_action_index(s, a)
        return (self.H[sa_index] * self.W).sum()

    def random_experience_sas(self, k=1):
        """
        Choose an experience sample uniformly at random (from unique
        experience list), and return it's s, a
        """
        unique_sas = np.unique(self.replay_buffer[:, 1:3], axis=0)
        idxs = np.random.choice(len(unique_sas), size=k, replace=True)
        exps = unique_sas[idxs]
        return exps

    def tiebreak_argmax(self, vals):
        return np.random.choice(np.flatnonzero(vals == vals.max()))

    def eps_greedy_policy(self, s, verbose=False):
        if self.env.reward_state(s) or s == self.env.terminal_state:
            return 0
        greedy = np.random.rand() > self.eps
        if greedy:
            sa_index = self.state_action_index(s, 0)
            qpi_all_actions = (self.H[sa_index:sa_index+4] * self.W).sum(axis=1)
            action = self.tiebreak_argmax(qpi_all_actions)
            if verbose:
                print("Q vals: %s, Action: %d" % (qpi_all_actions, action))
        else:
            action = self.env.random_action(s)
        return action

    def weighted_experience_samples(self, k=10):
        """
        Draw k samples from all unique (s, a)s, for each select a single
        transition from filtered buffer via exponential distribution (recency
        weighted).
        """
        experiences = self.replay_buffer
        random_sas = self.random_experience_sas(k=k)
        sample_ids = np.random.exponential(scale=1/self.exp_lambda, size=len(random_sas)).astype(int)
        experience_samples = []
        for sampled_id, from_sa in zip(sample_ids, random_sas):
            s, a = from_sa
            # Filter to only experiences from s, a
            exp_rows = (experiences[:, 1] == s) & (experiences[:, 2] == a)
            experiences_from_sa = experiences[exp_rows]
            n_exps = len(experiences_from_sa)
            # Recency weighting via sampled_id (Exponential over indexes)
            sampled_id = n_exps - np.clip(sampled_id, a_min=1, a_max=n_exps)
            experience_samples.append(experiences_from_sa[sampled_id])
        return np.array(experience_samples)

    def learn_offline(self, k=10):
        """
        Replay k transition samples (recency weighted) as per Eq 18
        """
        if not k:
            return
        samples = self.weighted_experience_samples(k=k)
        # Update state-action SR
        for t, s, a, s_prime, a_prime in samples:
            sa_prime_index = self.state_action_index(s_prime, 0)
            qs = (self.H[sa_prime_index:sa_prime_index+4] * self.W).sum(axis=1)
            if (qs == qs.max()).sum() > 1:
                # Ties, use a_prime (see https://github.com/evanrussek/Predictive-Representations-PLOS-CB-2017/blob/fb83671377d8ea0959fa421ef13f8f56d9dd65b2/agents/model_SRDYNA.m#L97)
                a_star = a_prime
            else:
                a_star = self.tiebreak_argmax(qs)
            sa_star_idx = self.state_action_index(s_prime, a_star)
            sa_idx = self.state_action_index(s, a)
            one_hot_sa = np.zeros(self.H.shape[0])
            one_hot_sa[sa_idx] = 1
            self.H[sa_idx] += self.alpha_sr * (one_hot_sa + self.gamma * self.H[sa_star_idx] - self.H[sa_idx])

    def learn(self, s, a, r, s_prime, a_prime, verbose=False):
        """
        Perform TD and TD-like updates to W and H.

        See eq 15 & 17, 8.
        """
        # Update H
        sa_idx = self.state_action_index(s, a)
        sa_prime_idx = self.state_action_index(s_prime, a_prime)
        one_hot_sa = np.zeros(self.H.shape[0])
        one_hot_sa[sa_idx] = 1
        self.H[sa_idx] += self.alpha_sr * (one_hot_sa + self.gamma * self.H[sa_prime_idx] - self.H[sa_idx])

        # Update value weights (W) from TD rule (Eq. 15)
        delta = r + self.gamma * self.q_pi(s_prime, a_prime) - self.q_pi(s, a)
        feature_rep = self.H[sa_idx]
        norm_feature_rep = feature_rep / (feature_rep ** 2).sum()
        w_update = self.alpha_w * delta * norm_feature_rep
        self.W += w_update

    def step(self, random_policy=False, verbose=False, learning=True):
        """
        Choose action
        Get next state from environment based on action
        Learn from s, a, r, s'
        Update state
        """
        if self.action is None:
            self.action = self.env.random_action(self.state)
        s_prime, r, done = self.env.successor(self.state, self.action)
        if random_policy:
            a_prime = self.env.random_action(s_prime)
        else:
            a_prime = self.eps_greedy_policy(s_prime, verbose=verbose)
        if learning:
            self.learn(self.state, self.action, r, s_prime, a_prime, verbose=verbose)
        if verbose:
            print("%s -> a:%d -> %s (r=%d)" % (self.state, self.action, s_prime, r))
        # Add to buffer
        exp = (self.t, self.state, self.action, s_prime, a_prime)
        self.replay_buffer = np.append(self.replay_buffer, [exp], axis=0)
        if learning:
            self.learn_offline(k=self.post_step_replays)  # 10 replay steps after each step
        self.last_state = self.state
        self.last_action = self.action
        self.state = s_prime
        self.action = a_prime
        self.t += 1
        self.ep_t += 1
        if done:
            self.terminate_episode()
        return done

    def corrected_value_map(self, mat_):
        """
        Reconstruct matrix swapping in single-action reward states
        for correct rendering.
        """
        mat = mat_.copy()
        for loc, ro in self.env.reward_locs.items():
            s = ro.get('s')
            sa = self.state_action_index(self.env.state_at_loc(loc, ignore_reward=True), 0)
            sa_goal = self.state_action_index(s, 0)
            mat[sa:sa+4] = mat[sa_goal]
        return mat

    def render_state_values(self, ax, fig=None, vmax=None):
        values = self.v_pi()
        ax.imshow(values.reshape(self.env.h, self.env.w),
                  origin='bottom',
                  cmap='Greys_r', vmin=0, vmax=vmax)
        ax.set_title("$V_{\\pi}$")
        ax.set_axis_off()

    def render_W(self, ax, fig=None, vmax=None):
        W = self.corrected_value_map(self.W)
        state_weights = W[:self.n_viz_sas()].reshape(-1, 4).max(axis=1)
        ax.imshow(state_weights.reshape(self.env.h, self.env.w),
                  origin='bottom',
                  cmap='Greys_r', vmin=0, vmax=vmax)
        ax.set_title("W (%.2f-%.2f)" % (state_weights.min(), state_weights.max()))
        ax.set_axis_off()

    def render_sr(self, s, ax, cmap='plasma', alpha=1.0):
        sa_idx = self.state_action_index(s, 0)
        state_sr = self.H[sa_idx:sa_idx+4].sum(axis=0)[:self.n_viz_sas()].reshape(-1, 4).max(axis=1)
        ax.imshow(state_sr.reshape(self.env.h, self.env.w), origin='bottom', alpha=alpha, cmap=cmap)
        loc = self.env.loc_at_state(s)
        ax.set_title("SR(%d, %d)" % (loc[0], loc[1]))
        ax.set_axis_off()

    def make_plots(self, sr_state=None):
        if sr_state is None:
            sr_state = self.state
        fig, axs = plt.subplots(1, 4, dpi=144)
        self.env.render(self, ax=axs[0], last_k_steps=self.ep_t)
        self.render_state_values(ax=axs[1], fig=fig)
        self.render_W(ax=axs[2], fig=fig)
        self.render_sr(sr_state, ax=axs[3])
        plt.show()

    def record_trials(self, title="recorded_trials", n_trial_per_loc=1,
                      start_locs=None, learning=False, max_steps=100):
        metadata = dict(title=title, artist='JG')
        writer = manimation.FFMpegFileWriter(fps=15, metadata=metadata)
        fig, axs = plt.subplots(1, 4, figsize=(7, 3))
        fig.tight_layout()

        self.eps = 0  # Fully greedy for recording

        with writer.saving(fig, "./out/%s.mp4" % title, 144):
            for sl in start_locs:
                for trial in range(n_trial_per_loc):
                    self.terminate_episode(reset_state=self.env.state_at_loc(sl))
                    done = False
                    steps = 0
                    while not done and steps < max_steps:
                        done = self.step(learning=learning)
                        self.env.render(self, ax=axs[0], last_k_steps=self.ep_t)
                        self.render_state_values(ax=axs[1], fig=fig)
                        self.render_W(ax=axs[2], fig=fig)
                        self.render_sr(self.state, ax=axs[3])
                        writer.grab_frame()
                        steps += 1
                        for ax in axs:
                            ax.clear()
