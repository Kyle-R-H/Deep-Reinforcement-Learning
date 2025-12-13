import random, collections
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym

import numpy as np

# To stop Numpy having a stroke
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ~~~~ Hyperparameters ~~~~
TRAIN = True
RENDER = not TRAIN

NUM_EPISODES = 200
MAX_STEPS = 200
BATCH_SIZE = 64
DISCOUNT = 0.99
LEARNING_RATE = 75e-4
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPS_START = 0.999
EPS_END = 0.01
EPS_DECAY = 0.997
TAU = 0.005

# ~~~~ Environment ~~~~
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("MountainCar-v0", render_mode="human" if RENDER else None)

# Seed everything
SEED = 21
random.seed(SEED)
torch.manual_seed(SEED)
env.reset(seed=SEED)
env.action_space.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



# ~~~~ Replay buffer ~~~~
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        s, a, r, s_, d = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(s_), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

# ~~~~ Normalise Function ~~~~
def normalize_state(state):
    low = env.observation_space.low
    high = env.observation_space.high
    return (state - low) / (high - low)

# ~~~~ Custom Rewards ~~~~
def custom_reward(state, next_state, env_reward):
    position, velocity = next_state

    # Base shaping
    modified_reward = 0.2 * (np.cos(np.deg2rad(position * 360)) + 2.0 * abs(velocity))
    modified_reward -= 0.9
    
    # Progress bonuses
    if position > 0.48:
        modified_reward += 10.0
    elif position > 0.40:
        modified_reward += 4.0
    elif position > 0.30:
        modified_reward += 1.0

    return float(np.clip(modified_reward, -50.0, 50.0))

def customStep(action):
    raw_next_state, env_reward, done, trunc, info = env.step(action)
    return normalize_state(raw_next_state), custom_reward(raw_next_state), done, trunc, info

def customReset(seed=None):
    if seed is not None:
        raw_state, info = env.reset(seed=seed)
    else:
        raw_state, info = env.reset()
    return normalize_state(raw_state), info

    
# ~~~~ Q-network ~~~~
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ~~~~ Agent ~~~~
class DQNAgent:
    def __init__(self, env):
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.online = QNet(obs_dim, n_actions).to(DEVICE)
        self.target = QNet(obs_dim, n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)

        self.replay = ReplayBuffer(BUFFER_SIZE)
        self.n_actions = n_actions
        self.steps = 0

    # Soft updating instead of hard updating for smoother transitions
    def soft_update(self):
        for t, s in zip(self.target.parameters(), self.online.parameters()):
            t.data.copy_(TAU * s.data + (1 - TAU) * t.data)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.online(state_v).argmax(dim=1).item())

    def learn(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        q_values = self.online(states).gather(1, actions.unsqueeze(1))

        # Double DQN
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * DISCOUNT * next_q

        # Huber loss
        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5)
        self.optimizer.step()

        self.steps += 1
        self.soft_update()
        return loss.item()
    
agent = DQNAgent(env)

# ~~ Pre-fill replay buffer ~~
state, _ = customReset(seed=SEED)
state = normalize_state(state)
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, env_reward, terminated, truncated, _ = customStep(action)
    next_state = normalize_state(next_state)
    done = terminated or truncated
    reward = custom_reward(state, next_state, env_reward)
    agent.replay.push(state, action, reward, next_state, done)
    state = next_state if not done else normalize_state(customReset()[0])

# Scout out the env
print("Pre-filled replay:", len(agent.replay))


# ~~~~ Plots ~~~~
def plot_training(rewards, losses, window=50, max_reward=1000):
    # Rewards plot
    rewards = np.array(rewards)
    rewards_history = np.clip(rewards, None, max_reward)

    if len(rewards) >= window:
        sma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        sma = np.clip(sma, None, max_reward)
    else:
        sma = None

    plt.figure()
    plt.title("Obtained Rewards")
    plt.plot(rewards_history, label="Raw Reward", color="#58A560")
    if sma is not None:
        plt.plot(sma, label=f"SMA {window}", color="#F08B17")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig("./reward_plot.png", dpi=600, bbox_inches="tight")
    plt.show()

    # Loss plot
    plt.figure()
    plt.title("Network Loss")
    plt.plot(losses, label="Loss", color="#9132BD")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./loss_plot.png", dpi=600, bbox_inches="tight")
    plt.show()


# ~~~~ Training loop ~~~~
def train():
    epsilon = EPS_START
    rewards_history = []
    loss_history = []

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = customReset()
        state = normalize_state(state)
        ep_reward = 0.0
        ep_losses = []
        ep_steps = 0

        for step in range(MAX_STEPS):
            action = agent.act(state, epsilon)
            next_state, env_reward, terminated, truncated, _ = customStep(action)
            next_state = normalize_state(next_state)

            done = terminated or truncated
            reward = custom_reward(state, next_state, env_reward)

            agent.replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_steps += 1

            if len(agent.replay) >= BATCH_SIZE:
                loss = agent.learn(BATCH_SIZE)
                ep_losses.append(loss)
                loss_history.append(loss)

            if done:
                break

        rewards_history.append(ep_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward {ep_reward:7.2f} | "
                f"Steps {ep_steps:3d} | "
                f"Epsilon {epsilon:.3f} | "
                f"Loss {loss_history:.3d}"
            )
    # Save model
    torch.save(agent.online.state_dict(), f"dqn_mountaincar_{NUM_EPISODES}.pth")
    print("Model saved.")
    plot_training(rewards_history, loss_history)


# ~~~~ Testing ~~~~
def test(max_episodes):
    for episode in range(1, max_episodes + 1):
        state, _ = customReset(seed=SEED)
        state = normalize_state(state)
        episode_reward = 0.0
        ep_steps = 0

        for _ in range(MAX_STEPS):
            action = agent.act(state, epsilon=0.0)
            next_state, env_reward, terminated, truncated, _ = customStep(action)
            next_state = normalize_state(next_state)

            done = terminated or truncated

            reward = custom_reward(state, next_state, env_reward)

            state = next_state
            episode_reward += reward
            ep_steps += 1
            if done:
                break
        print(
            f"Episode {episode:3d} | "
            f"Steps {ep_steps:3d} | "
            f"Return {episode_reward:7.2f}"
        )


# ~~~~ Main Runner ~~~~
if __name__ == "__main__":
    if TRAIN:
        train()
    else:
        model_path = f"dqn_mountaincar_{NUM_EPISODES}.pth"

        agent.online.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.online.eval()
        agent.target.load_state_dict(agent.online.state_dict())
        print(f"Loaded model from {model_path}")

        test(2)
