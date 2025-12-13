import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------ Config / Seeding ------------------ #
ENV_NAME = "MountainCar-v0"
RENDER = False
SEED = 21

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = gym.make(ENV_NAME, render_mode="human" if RENDER else None)
env.reset(seed=SEED)
env.action_space.seed(SEED)
try:
    env.observation_space.seed(SEED)
except Exception:
    pass
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ------------------ DQN Network ------------------ #
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 25)
        self.fc3 = nn.Linear(25, action_dim)
        # init weights (optional)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values (no activation)

# ------------------ DQN Agent ------------------ #
class DQNAgent:
    def __init__(self, state_dim, action_dim, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)

        # hyperparams
        self.epsilon = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 64
        self.lr = 0.001

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # store numpy 1D arrays for states
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # state is expected as 1D numpy array (obs_dim,)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)  # shape (1, obs_dim)
        with torch.no_grad():
            q_values = self.model(state_t)  # (1, action_dim)
        return int(q_values.argmax(dim=1).item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None  # nothing to learn yet

        minibatch = random.sample(self.memory, self.batch_size)

        # Convert lists of numpy arrays to a single numpy array first (fast)
        states_np = np.array([m[0] for m in minibatch], dtype=np.float32)      # (B, obs_dim)
        actions_np = np.array([m[1] for m in minibatch], dtype=np.int64)       # (B,)
        rewards_np = np.array([m[2] for m in minibatch], dtype=np.float32)     # (B,)
        next_states_np = np.array([m[3] for m in minibatch], dtype=np.float32) # (B, obs_dim)
        dones_np = np.array([m[4] for m in minibatch], dtype=np.float32)       # (B,)

        states = torch.from_numpy(states_np).to(self.device)           # (B, obs_dim)
        actions = torch.from_numpy(actions_np).to(self.device)         # (B,)
        rewards = torch.from_numpy(rewards_np).to(self.device)         # (B,)
        next_states = torch.from_numpy(next_states_np).to(self.device) # (B, obs_dim)
        dones = torch.from_numpy(dones_np).to(self.device)             # (B,)

        # current Q values for taken actions -> gather requires actions as (B,1)
        q_values_all = self.model(states)                              # (B, action_dim)
        q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # target calculation (using same network for next-values; could use target network)
        with torch.no_grad():
            next_q_all = self.model(next_states)                      # (B, action_dim)
            max_next_q_values = next_q_all.max(dim=1)[0]             # (B,)
            targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)  # (B,)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)    

        return loss.item()

# ------------------ Reward shaping ------------------ #
def get_reward(raw_state):
    # raw_state is environment observation (position, velocity)
    position = float(raw_state[0])
    if position >= 0.5:
        # reached goal
        return 10.0
    if position > -0.4:
        return float((1.0 + position) ** 2)
    return 0.0

# ------------------ Training function ------------------ #
def train_dqn(episodes=60, max_steps=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, device=device)
    scores = []

    for e in range(episodes):
        raw_state, _ = env.reset()
        state = np.array(raw_state, dtype=np.float32)   # keep as 1D array
        score = 0.0

        for step in range(max_steps):
            action = agent.act(state)
            if RENDER:
                env.render()
            raw_next_state, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            reward = get_reward(raw_next_state)
            next_state = np.array(raw_next_state, dtype=np.float32)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            loss = agent.replay()
            score += reward

            if done:
                print(f"episode: {e+1}/{episodes}, score: {score:.2f}, epsilon: {agent.epsilon:.3f}")
                break
        scores.append(score)
    return scores

# ------------------ Main ------------------ #
if __name__ == "__main__":
    episodes = 60
    scores = train_dqn(episodes=episodes, max_steps=1000)
    plt.plot(range(1, episodes + 1), scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
    env.close()