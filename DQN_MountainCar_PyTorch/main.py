import random, collections
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import gym

import numpy as np

# To stop Numpy having a stroke
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ~~~~ Seed & Device ~~~~
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ~~~~ Hyperparameters ~~~~
NUM_EPISODES = 400
MAX_STEPS = 200
BATCH_SIZE = 64
DISCOUNT = 0.99
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.997

TRAIN = True
RENDER = False


# ~~~~ Replay buffer ~~~~
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
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

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(
                0
            )
            return int(self.online(state_v).argmax(dim=1).item())

    def learn(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * DISCOUNT * next_q

        # Huber loss
        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()


# ~~~~ Environment ~~~~
env = gym.make("MountainCar-v0", render_mode="human" if RENDER else None)
agent = DQNAgent(env)


# ~~~~ Custom Rewards ~~~~
def custom_reward(state, next_state, env_reward):
    position, velocity = next_state

    # Base shaping
    modified_reward = 0.2 * (np.cos(np.deg2rad(position * 360)) + 2.0 * abs(velocity))

    # Progress bonuses
    if position > 0.48:
        modified_reward += 10.0
    elif position > 0.40:
        modified_reward += 4.0
    elif position > 0.30:
        modified_reward += 1.0

    modified_reward += env_reward  

    return float(np.clip(modified_reward, -50.0, 50.0))
    # return env_reward


# ~~ Pre-fill replay buffer ~~
state, _ = env.reset(seed=SEED)
state = normalize_state(state)
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, env_reward, terminated, truncated, _ = env.step(action)
    next_state = normalize_state(next_state)

    done = terminated or truncated
    reward = custom_reward(state, next_state, env_reward)
    agent.replay.push(state, action, reward, next_state, done)
    state = next_state if not done else env.reset()[0]

# Scout out the env
print("Pre-filled replay:", len(agent.replay))


# ~~~~ Plots ~~~~
def plot_training(rewards, losses, window=50, max_reward=100):
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
        plt.plot(
            # range(window - 1, len(rewards_history)),
            sma,
            label=f"SMA {window}",
            color="#F08B17",
        )
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
    step_history = []

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        state = normalize_state(state)
        ep_reward = 0.0
        ep_losses = []
        ep_steps = 0

        for step in range(MAX_STEPS):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
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

        if episode < 50:
            epsilon = 1.0
        else:
            epsilon = max(EPS_END, epsilon * EPS_DECAY)
        rewards_history.append(ep_reward)
        step_history.append(ep_steps)

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward {ep_reward:7.2f} | "
                f"Steps {ep_steps:3d} | "
                f"Reward Avg10 {np.mean(rewards_history[-10:]):7.2f} | "
                f"Epsilon {epsilon:.3f}"
            )
    # ~~~~ Save model ~~~~
    torch.save(agent.online.state_dict(), f"dqn_mountaincar_{NUM_EPISODES}.pth")
    print("Model saved.")
    plot_training(rewards_history, loss_history)


# ~~~~ Testing ~~~~
def test(max_episodes):
    returns = []
    steps_history = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=SEED)
        state = normalize_state(state)
        episode_reward = 0.0
        ep_steps = 0

        for _ in range(MAX_STEPS):
            action = agent.act(state, epsilon=0.0)  # greedy
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_state(next_state)

            done = terminated or truncated

            reward = custom_reward(state, next_state, env_reward)

            state = next_state
            episode_reward += reward
            ep_steps += 1

            if done:
                break

        returns.append(episode_reward)
        steps_history.append(ep_steps)

        print(
            f"Episode {episode:3d} | "
            f"Steps {ep_steps:3d} | "
            f"Return {episode_reward:8.2f}"
        )

    returns = np.array(returns)
    steps_history = np.array(steps_history)

    print(
        f"\nTest Summary - {max_episodes} episodes:\n"
        f"Mean Return: {returns.mean():.2f}\n"
        f"Mean Steps : {steps_history.mean():.1f}"
    )

    return returns, steps_history


# ~~~~ Main Runner ~~~~
if __name__ == "__main__":
    if TRAIN:
        print("Train")
        train()
    else:
        print("Test")
        episodes = 400
        model_path = f"dqn_mountaincar_{episodes}.pth"

        agent.online.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.online.eval()
        agent.target.load_state_dict(agent.online.state_dict())
        print(f"Loaded model from {model_path}")

        test(2)
