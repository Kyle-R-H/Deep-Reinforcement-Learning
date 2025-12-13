import random, collections
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
import numpy as np
import torch.nn.functional as F

# Compatibility
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ~~~~ Hyperparameters ~~~~
TRAIN = True
RENDER = not TRAIN

NUM_EPISODES = 800
MAX_STEPS = 200
BATCH_SIZE = 64
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
BUFFER_SIZE = 40000
MIN_REPLAY_SIZE = 1000
EPS_START = 0.999
EPS_END = 0.01
EPS_DECAY = 0.998
TAU = 0.005

# ~~~~ Environment ~~~~
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("MountainCar-v0", render_mode="human" if RENDER else None)

# Seed
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
        state, action , reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.stack(state), dtype=torch.float32, device=DEVICE),
            torch.tensor(action, dtype=torch.long, device=DEVICE),
            torch.tensor(reward, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(next_state), dtype=torch.float32, device=DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ~~~~ Normalise Function ~~~~
def normalize_state(state):
    low = env.observation_space.low
    high = env.observation_space.high
    return (state - low) / (high - low)


# ~~~~ Custom reward ~~~~
def custom_reward(next_state, env_reward=0.0):
    # next_state is a numpy array [position, velocity]
    position, velocity = next_state
    # base shaping (same idea as yours)
    modified_reward = 0.2 * (np.cos(np.deg2rad(position * 360)) + 2.0 * abs(velocity))
    modified_reward -= 0.9

    if position > 0.48:
        modified_reward += 10.0
    elif position > 0.40:
        modified_reward += 4.0
    elif position > 0.30:
        modified_reward += 1.0

    # clip to avoid huge spikes (you can tune this)
    return float(np.clip(modified_reward, -50.0, 50.0))


# ~~~~ Custom Step and reset ~~~~
def custom_step(action):
    raw_next_state, env_reward, terminated, truncated, info = env.step(action)
    mod_reward = custom_reward(raw_next_state, env_reward)
    norm_next_state = normalize_state(raw_next_state)
    return norm_next_state, mod_reward, terminated, truncated, info


def custom_reset(seed=None):
    if seed is not None:
        raw_state, info = env.reset(seed=seed)
    else:
        raw_state, info = env.reset()
    return normalize_state(raw_state), info


# ~~~~ Q-network ~~~~
class QNet(nn.Module):
    # def __init__(self, obs_dim, n_actions):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obs_dim, 24),
    #         nn.ReLU(),
    #         nn.Linear(24, 16),
    #         nn.ReLU(),
    #         nn.Linear(16, n_actions),
    #     )

    # def forward(self, x):
    #     return self.net(x)
    
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 25)
        self.fc2 = nn.Linear(25, 20)
        self.fc3 = nn.Linear(20, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
        actions_idx = actions.unsqueeze(1)          
        rewards = rewards.unsqueeze(1)              
        dones = dones.unsqueeze(1)                  

        q_values = self.online(states).gather(1, actions_idx) 

        # Double DQN target
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True) 
            next_q = self.target(next_states).gather(1, next_actions)           
            target_q = rewards + (1.0 - dones) * DISCOUNT * next_q              

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
state, _ = custom_reset(seed=SEED)
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = custom_step(action)
    done = terminated or truncated
    agent.replay.push(state, action, reward, next_state, done)
    if done:
        state, _ = custom_reset()   # start new episode in prefill
    else:
        state = next_state

# ~~~~ Plots ~~~~
def plot_training(rewards, losses, window=50, max_reward=1000):
    rewards = np.array(rewards)
    rewards_history = np.clip(rewards, None, max_reward)

    if len(rewards) >= window:
        sma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        sma = np.clip(sma, None, max_reward)
    else:
        sma = None

    plt.figure()
    plt.title("Obtained Rewards")
    plt.plot(rewards_history, label="Raw Reward")
    if sma is not None:
        plt.plot(sma, label=f"SMA {window}")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig("./reward_plot.png", dpi=600, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.title("Network Loss")
    plt.plot(losses, label="Loss")
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
        state, _ = custom_reset()
        ep_reward = 0.0
        ep_losses = []
        ep_steps = 0

        for step in range(MAX_STEPS):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = custom_step(action)
            done = terminated or truncated

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

        last_loss = loss_history[-1] if loss_history else 0.0
        if ep_steps < 200 or episode % 10 == 0:
        # if ep_steps < 200 or episode % 100 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward {ep_reward:7.2f} | "
                f"Steps {ep_steps:3d} | "
                f"Epsilon {epsilon:.3f} | "
                f"LastLoss {last_loss:.4f}"
            )

    torch.save(agent.online.state_dict(), f"dqn_mountaincar_{NUM_EPISODES}.pth")
    print("Model saved.")
    plot_training(rewards_history, loss_history)


# ~~~~ Testing ~~~~
def test(max_episodes):
    for episode in range(1, max_episodes + 1):
        state, _ = custom_reset(seed=SEED)
        episode_reward = 0.0
        ep_steps = 0

        for _ in range(MAX_STEPS):
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = custom_step(action)
            done = terminated or truncated

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
