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
q = 10 # number of time history required. 
nx = 4 #dim of state
nu = 1 # dim of control
nz = 2 # dim of measurements
nZ = q*nz + (q-1)*nu #dim of information state

def make_env(env_id, render_bool):

    if render_bool:

        env = gym.make('InvertedPendulum-v4',render_mode = "human")
    else:
        env = gym.make('InvertedPendulum-v4')

    min_action = -20
    max_action = 20
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (nZ,), np.float64)
    env.reset()

    return env

def information_state(prev_info_state, next_obs, control):

    info_state = np.zeros((nZ))
    info_state[0:2] = next_obs[0:2]
    info_state[2:q*nz] = prev_info_state[0:(q-1)*nz]

    info_state[q*nz:q*nz+nu] = control
    info_state[q*nz+nu:nZ] = prev_info_state[q*nz:q*nz + (q-2)*nu]
    
    return info_state

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
    checkpoint = torch.load("/home/naveed/Documents/RL/naveed_codes/runs/test_po/carpole_test_po_q_10.cleanrl_model")
    actor.load_state_dict(checkpoint[0])
    qf1.load_state_dict(checkpoint[1])

    actor.eval()
    qf1.eval() 

    obs, _ = env.reset(seed=given_seed)
    prev_actions = np.array([0])
    info_state = np.zeros((nZ))
    
    #initial transient
    for transient_step in range(q):
        actions = 0.001*np.array(env.action_space.sample())
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        info_state = information_state(info_state, next_obs, actions)

        #print('info_state = ', info_state, ' actions=', actions)

    for global_step in range(total_timesteps):

        prev_info_state = info_state

        with torch.no_grad():
            actions = actor(torch.Tensor(info_state).to(device))
            cost_to_go = -qf1(torch.Tensor(info_state).to(device), actions).item()
            actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        info_state = information_state(info_state, next_obs, actions)
        
        if terminations:
            next_obs, _ = env.reset(seed=given_seed)
        #env.render()
        
        print("observation:", next_obs, " action:", actions, ' CTG=', cost_to_go)
        
        obs = next_obs
        prev_actions = actions

    env.close()