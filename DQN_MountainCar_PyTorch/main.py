# Classic Control
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyperparams = dict(
    env_id="MountainCar-v0", 
    render_mode="rgb_array", 
    goal_vel=0,
    env_seed = 21,
    
    epsilon_start = 0.999,
    epsilon_end = 0.01,
    decay_rate = 0.997,
    
    max_steps = 200,
    batch_size = 64,
    replay_max = 2000,
    
    log_rate = 1000
)

env = gym.make(
    hyperparams["environment_id"],
    render_mode=hyperparams["render_mode"],
    goal_velocity=hyperparams["goal_vel"],
)

env._np_random_seed(hyperparams["env_seed"])

def Rendering():
    print()


def Network():
    print()


def Agent():
    print()


def policy():
    print()


def Reward():
    print()


def loss():
    print()


def optimiser():
    print()


def train():
    print()


def test():
    print()


# ==== Main Runner ====
if __name__ == "__main__":
    print()
