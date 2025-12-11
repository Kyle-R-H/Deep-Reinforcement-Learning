# import os
# import gc
import time
import random
# import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
# import pygame

# Seed + Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
#                 1. Replay Memory (Optimised)
# =====================================================
class ReplayMemory:
    def __init__(self, capacity, obs_shape):
        self.capacity = int(capacity)
        self.pos = 0
        self.size = 0

        self.s = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)

    def store(self, s, a, s2, r, d):
        idx = self.pos
        self.s[idx] = s
        self.a[idx] = a
        self.s2[idx] = s2
        self.r[idx] = r
        self.d[idx] = d

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        s = torch.tensor(self.s[idx], dtype=torch.float32, device=device)
        a = torch.tensor(self.a[idx], dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor(self.r[idx], dtype=torch.float32, device=device).unsqueeze(1)
        s2 = torch.tensor(self.s2[idx], dtype=torch.float32, device=device)
        d = torch.tensor(
            self.d[idx].astype(np.float32), dtype=torch.float32, device=device
        ).unsqueeze(1)

        return s, a, r, s2, d

    def __len__(self):
        return self.size


# =====================================================
#                 2. DQN Network
# =====================================================
class DQN_Network(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.net(x)


# =====================================================
#                 3. DQN Agent (Optimised)
# =====================================================
class DQN_Agent:
    def __init__(
        self,
        obs_dim,
        num_actions,
        memory_capacity,
        lr=1e-3,
        gamma=0.99,
        grad_clip=5.0,
        double=True,
    ):

        self.gamma = gamma
        self.grad_clip = grad_clip
        self.double = double

        self.online = DQN_Network(obs_dim, num_actions).to(device)
        self.target = DQN_Network(obs_dim, num_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.optim = optim.Adam(self.online.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity, (obs_dim,))

        # epsilon schedule parameters
        self.eps_start = 0.999
        self.eps_end = 0.01
        self.eps_decay_steps = 50000
        self.total_steps = 0

        self.loss_history = []

    # ---------------------- epsilon schedule ----------------------
    def epsilon(self):
        frac = min(self.total_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    # ---------------------- action selection ----------------------
    def select_action(self, state):
        if random.random() < self.epsilon():
            return random.randrange(self.online.net[-1].out_features)

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(state).argmax().item())

    # --------------------- learning step --------------------------
    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        s, a, r, s2, d = self.memory.sample(batch_size)

        # predicted Q(s,a)
        q_sa = self.online(s).gather(1, a)

        # ------------------ compute targets ------------------------
        with torch.no_grad():
            if self.double:
                # online selects best action
                best_a = self.online(s2).argmax(dim=1, keepdim=True)
                # target evaluates it
                q_next = self.target(s2).gather(1, best_a)
            else:
                q_next = self.target(s2).max(dim=1, keepdim=True)[0]

            y = r + self.gamma * (1 - d) * q_next

        # ------------------ loss (Huber) ---------------------------
        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.loss_history.append(loss.item())

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

    # ------------------ target network sync ------------------------
    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())


# =====================================================
#                 4. Environment Wrappers (unchanged)
# =====================================================
class observation_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.min = env.observation_space.low
        self.max = env.observation_space.high

    def observation(self, state):
        return (state - self.min) / (self.max - self.min)


class reward_wrapper(gym.RewardWrapper):
    def reward(self, state):
        pos, vel = state
        vel = np.interp(vel, [0, 1], [-0.5, 0.5])
        degree = pos * 360
        rad = np.deg2rad(degree)

        reward = 0.2 * (np.cos(rad) + 2 * abs(vel)) - 0.5

        if pos > 0.98:
            reward += 20
        elif pos > 0.92:
            reward += 10
        elif pos > 0.82:
            reward += 6
        elif pos > 0.65:
            reward += 1 - np.exp(-2 * pos)

        if vel > 0.3 and pos > 0.5084:
            reward += 1 + 2 * pos

        return float(reward)


class step_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_wrap = observation_wrapper(env)
        self.r_wrap = reward_wrapper(env)

    def step(self, action):
        s, r, term, trunc, info = self.env.step(action)
        s2 = self.obs_wrap.observation(s)
        r2 = self.r_wrap.reward(s2)
        return s2, r2, term, trunc, info

    def reset(self, seed=None):
        s, info = self.env.reset(seed=seed)
        return self.obs_wrap.observation(s), info


# =====================================================
#                 5. Model Trainer
# =====================================================
class Model_TrainTest:
    def __init__(self, train=True, render=False):
        self.train_mode = train
        self.render = render

        self.batch_size = 64
        self.max_episodes = 200
        self.max_steps = 200
        self.update_every = 1000
        self.memory_capacity = 125000

        env = gym.make(
            "MountainCar-v0",
            render_mode="human" if render else None,
            max_episode_steps=self.max_steps,
        )

        env = step_wrapper(env)
        self.env = env

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.agent = DQN_Agent(
            obs_dim=obs_dim,
            num_actions=n_actions,
            memory_capacity=self.memory_capacity,
            lr=7.5e-4,
            gamma=0.96,
            double=True,
        )

        self.rewards = []

    def train(self):
        global_step = 0

        for ep in range(1, self.max_episodes + 1):
            s, _ = self.env.reset(seed=seed)
            done = False
            ep_reward = 0
            steps = 0

            while not done:
                global_step += 1
                self.agent.total_steps = global_step

                a = self.agent.select_action(s)
                s2, r, term, trunc, _ = self.env.step(a)
                done = term or trunc

                self.agent.memory.store(s, a, s2, r, done)
                self.agent.learn(self.batch_size)

                if global_step % self.update_every == 0:
                    self.agent.update_target()

                s = s2
                ep_reward += r
                steps += 1

                if self.render:
                    time.sleep(1 / 60)

            self.rewards.append(ep_reward)
            print(
                f"Episode {ep} | Reward: {ep_reward:.2f} | Epsilon: {self.agent.epsilon():.3f}"
            )

        self.plot()

    def test(self, n=5):
        self.agent.online.load_state_dict(torch.load("./dqn.pth"))

        for ep in range(n):
            s, _ = self.env.reset(seed=ep)
            done = False
            total_reward = 0
            steps = 0

            while not done:
                a = int(
                    torch.argmax(
                        self.agent.online(
                            torch.tensor(
                                s, dtype=torch.float32, device=device
                            ).unsqueeze(0)
                        )
                    )
                )
                s, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                total_reward += r
                steps += 1

            print(f"[TEST] Episode {ep+1} | Reward: {total_reward:.2f}")

    def plot(self):
        plt.plot(self.rewards, label="Reward")
        if len(self.rewards) > 20:
            sma = np.convolve(self.rewards, np.ones(20) / 20, mode="valid")
            plt.plot(range(19, 19 + len(sma)), sma, label="SMA20")
        plt.legend()
        plt.show()


# =====================================================
#                6. Main
# =====================================================
if __name__ == "__main__":
    train = False
    trainer = Model_TrainTest(train=train, render= not train)
    trainer.train()
