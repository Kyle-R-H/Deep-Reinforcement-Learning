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
TRAIN = False
RENDER = not TRAIN

NUM_AGENTS = 4
NUM_EPISODES = 1000
MAX_STEPS = 200
BATCH_SIZE = 64
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
# LR_DECAY = 0.001
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 5000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TAU = 0.001

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
def Observe_state(state):
    low = env.observation_space.low
    high = env.observation_space.high
    # print(f"State: {state} | Low: {low} | High: {high}")
    output = (state - low) / (high - low)
    # print(f"Out: {output}")
    return output


# ~~~~ Custom reward ~~~~
def custom_reward(next_state, env_reward=0.0):
    position, velocity = next_state
    # Env reward is -1 for each step
    reward = env_reward
    # Velocity
    reward += 5 * abs(velocity)
    if position >= 0.5:
        reward += 50

    # print(f"Env Reward: {env_reward} | Mod Reward: {reward}")
    return reward


# ~~~~ Custom Step and reset ~~~~
def custom_step(action):
    raw_next_state, env_reward, terminated, truncated, info = env.step(action)
    mod_reward = custom_reward(raw_next_state, env_reward)
    norm_next_state = Observe_state(raw_next_state)
    return norm_next_state, mod_reward, terminated, truncated, info

def custom_reset(seed=None):
    raw_state, info = env.reset(seed=seed)
    return Observe_state(raw_state), info


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
    def __init__(self, env, replay_buffer):
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.online = QNet(obs_dim, n_actions).to(DEVICE)
        self.target = QNet(obs_dim, n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=LEARNING_RATE,
            # weight_decay=LR_DECAY
        )

        self.replay = replay_buffer
        self.steps = 0

    def soft_update(self):
        for t, s in zip(self.target.parameters(), self.online.parameters()):
            t.data.copy_(TAU * s.data + (1 - TAU) * t.data)

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

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5)
        self.optimizer.step()

        self.steps += 1
        self.soft_update()
        
        # # Hard update - Didnt work as good as soft update when testing
        # if self.steps % 1000 == 0:
        #     self.target.load_state_dict(self.online.state_dict())
        
        return loss.item()

# ~~~~ Multi-Agent Act ~~~~
def multi_agent_act(agents, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    with torch.no_grad():
        state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        qs = [agent.online(state_v) for agent in agents]
        mean_q = torch.mean(torch.stack(qs), dim=0)
        return int(mean_q.argmax(dim=1).item())


# ~~ Pre-fill replay buffer ~~
def buffer_prefill():
    state, _ = custom_reset(seed=SEED)
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = custom_step(action)
        shared_replay.push(state, action, reward, next_state, terminated)
        if terminated:
            state, _ = custom_reset()
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
    plt.title(f"LR: {LEARNING_RATE} | Buf Size: {BUFFER_SIZE} | Epsi Decay: {EPS_DECAY}")
    plt.plot(rewards_history, label="Raw Reward")
    if sma is not None:
        plt.plot(sma, label=f"SMA {window}")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig("./reward_plot.png", dpi=600, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.title(f"LR: {LEARNING_RATE} | Buf Size: {BUFFER_SIZE} | Epsi Decay: {EPS_DECAY}")
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
        ep_steps = 0

        for step in range(MAX_STEPS):
            action = multi_agent_act(agents, state, epsilon)
            next_state, reward, terminated, truncated, _ = custom_step(action)

            shared_replay.push(state, action, reward, next_state, terminated)
            state = next_state
            ep_reward += reward
            ep_steps += 1

            # 4 is learning delay
            if step % 4 == 0 and len(shared_replay) >= BATCH_SIZE:
                for agent in agents:
                    loss = agent.learn(BATCH_SIZE)
                    loss_history.append(loss)

            if terminated:
                break

        rewards_history.append(ep_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep_steps < 200 or episode % 10 == 0:
        # if ep_steps < 200 or episode % 100 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward {ep_reward:7.2f} | "
                f"Steps {ep_steps:3d} | "
                f"Epsilon {epsilon:.3f} | "
            )

    torch.save(agents[0].online.state_dict(), f"dqn_mountaincar_{NUM_EPISODES}.pth")
    print("Model saved.")
    plot_training(rewards_history, loss_history)


# ~~~~ Testing ~~~~
def test(episodes):
    for episode in range(episodes):
        state, _ = custom_reset(seed=SEED)
        episode_reward = 0.0
        ep_steps = 0

        for _ in range(MAX_STEPS):
            action = multi_agent_act(agents, state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = custom_step(action)

            state = next_state
            episode_reward += reward
            ep_steps += 1
            if terminated:
                break
        print(
            f"Episode {episode+1:3d} | "
            f"Steps {ep_steps:3d} | "
            f"Return {episode_reward:7.2f}"
        )


# ~~~~ Main Runner ~~~~
if __name__ == "__main__":
    shared_replay = ReplayBuffer(BUFFER_SIZE)
    agents = [DQNAgent(env, shared_replay) for _ in range(NUM_AGENTS)]


    if TRAIN:
        buffer_prefill()
        train()
    else:
        for agent in agents:
            agent.online.load_state_dict(torch.load(f"dqn_mountaincar_{NUM_EPISODES}.pth", map_location=DEVICE))
            agent.target.load_state_dict(agent.online.state_dict())
        test(2)
