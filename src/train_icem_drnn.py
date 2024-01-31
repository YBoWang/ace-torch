import datetime
import warnings
warnings.filterwarnings('ignore')
import os

# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from envs.env import make_env
# from envs.maze_env import make_maze_env
from algorithm.ace_icem_similarity_drnn import AceDRNN
from algorithm.helper import Episode, ReplayBuffer
import logger

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        hidden = None
        while not done:
            if t == 0 or t % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, _ = agent.plan(obs, hidden, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
    return np.nanmean(episode_rewards)


def evaluate_discounted_mc(env, agent, step, num_episodes=10, gamma=0.99, n_mc_cutoff=85):
    final_mc_list = np.zeros(0)
    final_obs_list, final_act_list = [], []
    episode_rewards = []
    total_mc_samples = num_episodes * n_mc_cutoff
    while final_mc_list.shape[0] < total_mc_samples:
        obs, done, r, ep_len, ep_return = env.reset(), False, 0, 0, 0
        obs_list, act_list, reward_list = [], [], []
        while not done:
            hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, _ = agent.plan(obs, hidden, eval_mode=True, step=step, t0=ep_len == 0)
            obs_list.append(obs)
            act_list.append(action.cpu().numpy())
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_return += reward
            ep_len += 1
            reward_list.append(reward)
        discount_return_list = np.zeros(ep_len)
        for i in range(ep_len - 1, -1, -1):
            if i == ep_len - 1:
                discount_return_list[i] = reward_list[i]
            else:
                discount_return_list[i] = reward_list[i] + gamma * discount_return_list[i + 1]

        final_mc_list = np.concatenate((final_mc_list, discount_return_list[:n_mc_cutoff]))
        final_obs_list.extend(obs_list[:n_mc_cutoff])
        final_act_list.extend(act_list[:n_mc_cutoff])
        episode_rewards.append(ep_return)

    obs_tensor = torch.tensor(final_obs_list, dtype=torch.float32, device='cuda')
    acts_tensor = torch.tensor(final_act_list, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        zs = agent.model.h(obs_tensor)
        q_pred = torch.min(*agent.model.Q(zs, acts_tensor))
    q_pred = q_pred.cpu().numpy()
    bias = q_pred - final_mc_list
    final_mc_list_normalize_base = final_mc_list.copy()
    final_mc_list_normalize_base = np.abs(final_mc_list_normalize_base)
    final_mc_list_normalize_base[final_mc_list_normalize_base < 10] = 10
    normalized_bias_per_state = bias / final_mc_list_normalize_base
    bias_mean = np.mean(normalized_bias_per_state)
    bias_std = np.std(normalized_bias_per_state)
    return {
        'q_bias_mean': bias_mean,
        'q_bias_std': bias_std,
        'episode_reward': np.nanmean(episode_rewards),
    }


def evaluate_pi(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0)
            action = agent.model.pi(agent.model.h(obs))
            obs, reward, done, _ = env.step(action.squeeze().detach().cpu().numpy())
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
    return np.nanmean(episode_rewards)


def train(cfg):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_env(cfg), AceDRNN(cfg), ReplayBuffer(cfg, latent_plan=True)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # Collect trajectory
        interaction_start_time = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        total_train_step = step
        external_reward_mean_list = []
        current_std_mean_list = []
        while not episode.done:
            # reset the hidden state for gru every cfg.horizon step.
            hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, plan_metrics = agent.plan(obs, hidden, step=step, t0=episode.first)
            external_reward_mean_list.append(plan_metrics['external_reward_mean'])
            current_std_mean_list.append(plan_metrics['current_std'])
            obs, reward, done, _ = env.step(action.detach().cpu().numpy())
            episode += (obs, action, reward, done)
            total_train_step += 1
        assert len(episode) == cfg.episode_length
        buffer += episode

        interaction_time = time.time() - interaction_start_time

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))
        iter_time = time.time() - interaction_start_time

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        ep_len = cfg.episode_length * cfg.action_repeat
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'explorer_fps': np.clip(int(ep_len / interaction_time), 0, 500),
            'fps': np.clip(int(ep_len / iter_time), 0, 250),
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'external_reward_mean': np.mean(external_reward_mean_list),
            'current_std': np.mean(current_std_mean_list), }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics.update(evaluate_discounted_mc(env, agent, step))
            # common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
            L.log(common_metrics, category='eval')
        # print('planning time {0}, training time {1}, iteration time {2}'.format(planning_time,
        #                                                                         training_time, iteration_time))

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
