import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
from distutils.util import strtobool
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stable_baselines3.common.buffers import ReplayBuffer

ENV_NAME = 'InvertedPendulum-v4'

def make_env(env_id, render_bool):

    if render_bool:

        env = gym.make('InvertedPendulum-v4',render_mode = "human")
    else:
        env = gym.make('InvertedPendulum-v4')

    min_action = -20
    max_action = 20
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

if __name__ == "__main__":

    given_seed = 42
    buffer_size = int(1e6)
    batch_size = 256
    total_timesteps = 2000
    learning_starts = 25e3
    exploration_noise = 0.1
    policy_frequency = 2
    tau = 0.005
    gamma = 0.99
    learning_rate = 3e-4

    exp_name = 'carpole_test'
    run_name = 'test'
    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}");

    env = make_env(ENV_NAME, render_bool = True)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    checkpoint = torch.load("/home/naveed/Documents/RL/naveed_codes/runs/test/carpole_test.cleanrl_model")
    actor.load_state_dict(checkpoint[0])
    qf1.load_state_dict(checkpoint[1])

    actor.eval()
    qf1.eval() 

    obs, _ = env.reset(seed=given_seed)

    for global_step in range(total_timesteps):
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            cost_to_go = -qf1(torch.Tensor(obs).to(device), actions).view(-1)
            actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        if terminations:
            obs, _ = env.reset(seed=given_seed)
        #env.render()
        
        print("observation:", next_obs, " action:", actions, ' CTG=', cost_to_go)
        
        obs = next_obs

    env.close()