import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import random
import time

ENV_NAME = 'InvertedPendulum-v4'
exp_name = 'ppo_cartpole_1M'
run_name = 'ppo'

def make_env(env_id, render_bool, record_video=False):

    if record_video:
        env = gym.make('InvertedPendulum-v4',render_mode = "rgb_array")
        env = gym.wrappers.RecordVideo(env, f"../videos/ppo")

    elif render_bool: 
        env = gym.make('InvertedPendulum-v4',render_mode = "human")

    else:
        env = gym.make('InvertedPendulum-v4')

    min_action = -30
    max_action = 30
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.squeeze() 
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # Compute log probability and entropy for the action
        log_prob = probs.log_prob(action).sum()  # Sum over action dimensions (if >1)
        entropy = probs.entropy().sum()  # Sum over action dimensions (if >1)

        # Return action, log probability, entropy, and critic value
        return action, log_prob, entropy, self.critic(x)  # Critic(x) returns value

def reward_function(observation, action):
    diag_q = [1,10,1,1]; 
    r = 1;
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)

    return -cost

if __name__ == "__main__":

    given_seed = 1
    total_timesteps = 100
    gamma = 0.99

    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}");

    env = make_env(ENV_NAME, render_bool = True, record_video=False)

    agent = Agent(env).to(device)

    checkpoint = torch.load(f"../runs/{run_name}/{exp_name}.pth")
    agent.load_state_dict(checkpoint)
    
    agent.eval()

    obs, _ = env.reset(seed=10)
    q_pos = np.array([0,np.pi])
    q_vel = np.array([0,0])
    env.unwrapped.set_state(q_pos, q_vel)
    obs[0:2] = q_pos
    obs[2:] = q_vel

    cost = 0

    action_vec = []


    for global_step in range(total_timesteps):
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs).to(device))
            
            action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            action_vec.append(action[0])
            

        rewards = reward_function(obs, action)
        cost -=rewards
        next_obs, rewards, terminations, truncations, infos = env.step(action)
        
        if terminations:
            obs, _ = env.reset()
            # env.set_state(q_pos, q_vel)
            # obs, rewards, terminations, truncations, infos = env.step(0.0)
            
        env.render()
        time.sleep(0.1)
        print("observation:", obs, " action:", action, ' CTG=', value)
        
        obs = next_obs
    
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs).to(device))
        action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
    print("observation:", obs, " action:", action, ' CTG=', value)
    print("actions vec =", action_vec)
    #save_actions_to_csv(action_vec, filename="actions_sac_horizon_30.csv")

    print(f"Final cost = {cost}")
    env.close()