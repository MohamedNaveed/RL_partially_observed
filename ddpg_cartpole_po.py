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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
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

def reward_function(observation, action):
    diag_q = [1,10,1,1]; 
    r = 1;
    #print("observation:", observation)
    #print("observation:", observation[0,1])
    reward = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)

    return -reward

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

if __name__ == "__main__":

    given_seed = 1
    buffer_size = int(1e6)
    batch_size = 256
    total_timesteps = 100000 #default = 1000000
    learning_starts = 25000 #default = 25e3
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

    #reward function parameters
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}");

    env = make_env(ENV_NAME, render_bool = False)
    
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    # load pretrained model.
    checkpoint = torch.load("/home/naveed/Documents/RL/naveed_codes/runs/test/carpole_test.cleanrl_model")
    actor.load_state_dict(checkpoint[0])
    qf1.load_state_dict(checkpoint[1])

    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=given_seed)
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < learning_starts:
            actions = np.array(env.action_space.sample())
            
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * exploration_noise)
                actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        rewards = reward_function(next_obs, actions)
        #print('step=', global_step, ' actions=', actions, ' rewards=', rewards,\
        #      ' obs=', next_obs, ' termination=', terminations, ' trunctions=', truncations)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                #print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        # if truncations:
        #     real_next_os = infos["final_observation"]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            data = rb.sample(batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            
            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                print('step=', global_step, ' rewards=', rewards, ' qf1_loss = ', qf1_loss.item(), \
                      ' actor_loss = ', actor_loss.item(), ' observations=', obs, ' action=', actions)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if global_step % 100 == 0:
                
                print("SPS:", int(global_step / (time.time() - start_time)))


        if abs(next_obs[0])>= 10:
            print('resetting')
            obs, _ = env.reset(seed=given_seed)


    save_model = True
    if save_model:
        model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

    env.close()
    


