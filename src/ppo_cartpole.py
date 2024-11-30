import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import random
import time
import csv

ENV_NAME = 'InvertedPendulum-v4'
save_model = True
csv_file = 'ppo_cartpole_output_v4_1M.csv' #csv file to store training progress
exp_name = 'ppo_cartpole_1M'
run_name = 'ppo'

def make_env(env_id, render_bool):

    if render_bool:

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

def reward_function(observation, action):
    
    diag_q = [1,10,1,1] 
    r = 1
    
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)
    
    return -cost

def write_data_csv(data):
    

    # Write the data to a CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writerow(['step', 'cost', 'loss', 'observations', 'action'])
        
        # Write the data
        writer.writerow(data)

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

if __name__ == "__main__":

    given_seed = 1
    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    learning_rate = 3e-4
    num_steps = 2048
    num_envs = 1
    batch_size = int(num_envs * num_steps)
    num_minibatches = 32
    minibatch_size = int(batch_size // num_minibatches)
    total_timesteps = 1000000
    episode_length = 100
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    vf_coef = 0.5
    ent_coef = 0.0
    max_grad_norm = 0.5
    target_kl = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    #env setup 
    env = make_env(ENV_NAME, render_bool = False)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_envs) + env.observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + env.action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=given_seed)
    print(f'initial state = {next_obs}')
    next_obs = torch.Tensor(next_obs).to(device)
    if next_obs.dim() == len(env.observation_space.shape):  # Single environment
        next_obs = next_obs.unsqueeze(0)  # Add batch dimension for single-env
    
    next_done = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    num_updates = total_timesteps // batch_size

    episode_count = 0
    cost = 0
    episode_t = 0 
    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                action = action.clip(torch.from_numpy(env.action_space.low).to(device), 
                                     torch.from_numpy(env.action_space.high).to(device))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            
            next_obs, reward, terminations, truncations, infos = env.step(action.cpu().numpy().squeeze())
            reward = reward_function(next_obs, action.cpu().numpy().squeeze())
            cost -=reward

            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            # If 'done' is boolean or list/array of booleans, convert properly
            next_done = torch.tensor(done, dtype=torch.float32).to(device)
            if isinstance(next_done, torch.Tensor) and next_done.dim() == 0:  # Single environment
                next_done = next_done.unsqueeze(0)  # Add batch dimension
            
            episode_t += 1
            if abs(next_obs[0])>= 10 or (episode_t == episode_length):
                print(f'resetting at step {episode_t}')
                episode_count += 1
                if episode_count % 100 == 0:
                    write_data = [global_step, cost, loss.item(), obs, actions]
                    write_data_csv(write_data)
                
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                episode_t = 0
                print(f'Cost = {cost}')
                cost = 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

       
        print(f'update step= {update} loss = {loss.item()}')

    if save_model:
        import os
        os.makedirs(f"../runs/{run_name}", exist_ok=True)
        model_path = f"../runs/{run_name}/{exp_name}.pth"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.ppo_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     ENV_NAME,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=Agent,
        #     device=device,
        #     gamma=gamma,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        
    env.close()
    
