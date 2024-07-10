import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np

#env = gym.make("CartPole-v1", render_mode = "human") #"CartPole-v1" | "Swimmer-v4"
env = gym.make('InvertedPendulum-v4',render_mode = "human")
min_action = -20
max_action = 20
env = RescaleAction(env, min_action=min_action, max_action=max_action)
obs, _ = env.reset()
print('obs = ', obs.dtype)
#print('env.observation_space=', env.observation_space, ' env.action_space = ', env.action_space)

#new_env_obs_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float64)
#new_env_action_space = gym.spaces.Box(-20, 20, (1,), np.float32)
#print('new_env_obs_space=', new_env_obs_space, ' new_env_action_space = ', new_env_action_space)

#env.observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float64)

#print('env.observation_space=', env.observation_space,)
q_pos = np.array([0,np.pi])
q_vel = np.array([0,0])
#env.set_state(q_pos, q_vel)

for i in range(300):
    action =  env.action_space.sample()
    print("action:", action)
    
    observation, reward, termination, truncated, info = env.step(action)
    env.render()
    print("step:", i, "observation:", observation," reward:", reward, " termination", termination, " truncated:", truncated)
    if abs(observation[0])>= 10:
        observation, info = env.reset()
    #     env.set_state(q_pos, q_vel)

env.close()