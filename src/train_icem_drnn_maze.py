import warnings
warnings.filterwarnings('ignore')
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
import h5py
import gzip
import pickle

gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from envs.maze_env import make_maze_env
from algorithm.ace_icem_similarity_drnn import TdICemSimDssm
from algorithm.helper import Episode, ReplayBuffer
import logger

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }


def append_data(data, s, a, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())


def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def get_occupancy_idx(sim_data, free_grid_centers):
    qpos = sim_data.qpos.ravel()
    for free_grid_center in free_grid_centers:
        if np.allclose(qpos, free_grid_center, atol=0.5, rtol=0.0):
            return free_grid_center


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        hidden = None
        while not done:
            if t == 0 or t % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, _ = agent.plan(obs, hidden, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            if video: video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)


def evaluate_pi(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0)
            action = agent.model.pi(agent.model.h(obs))
            obs, reward, done, _ = env.step(action.squeeze().detach().cpu().numpy())
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video:
            video.save(env_step)
    return np.nanmean(episode_rewards)


def train(cfg):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_maze_env(cfg), TdICemSimDssm(cfg), ReplayBuffer(cfg, latent_plan=True)

    # log for the occupancy map of the exploration policy
    num_free_grids = len(env.empty_and_goal_locations)
    free_grids_centers = np.array(env.empty_and_goal_locations) - 0.5
    offline_dataset, eval_dataset = reset_data(), reset_data()
    # prev_end_point = None
    grid_occupancy_list = []
    eval_grid_occupancy_list = []

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        grids_visited_counts = np.zeros(env.maze_arr.shape)
        hidden = None
        total_train_step = step
        external_reward_mean_list = []
        current_std_mean_list = []
        while not episode.done:
            # # planing based exploration policy
            # # reset the hidden state for gru every cfg.horizon step.
            if episode.first or total_train_step % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, plan_metrics = agent.plan(obs, hidden, step=step, t0=episode.first)
            external_reward_mean_list.append(plan_metrics['external_reward_mean'])
            current_std_mean_list.append(plan_metrics['current_std'])

            # # off-policy exploration policy
            # if step < cfg.seed_steps:
            #     action = torch.empty(cfg.action_dim, dtype=torch.float32, device=cfg.device).uniform_(-1, 1)
            # else:
            #     obs = torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0)
            #     action = agent.model.pi(agent.model.h(obs), std=0.1).squeeze().detach()

            obs, reward, done, _ = env.step(action.cpu().numpy())
            if cfg.pure_intrinsic_reward:
                reward = 0.0
            append_data(offline_dataset, obs, action.cpu().numpy(), env.goal_locations[0],
                        done, env.sim.data)
            free_grid_center = get_occupancy_idx(env.sim.data, free_grids_centers)
            if free_grid_center is not None:
                visited_grid = free_grid_center - 0.5
                grids_visited_counts[int(visited_grid[0]), int(visited_grid[1])] += 1
            episode += (obs, action, reward, done)
            total_train_step += 1
        assert len(episode) == cfg.episode_length
        buffer += episode
        grid_occupancy_list.append(grids_visited_counts.reshape(-1))
        num_visited_grids = np.count_nonzero(grids_visited_counts)
        epoch_state_coverage = num_visited_grids / num_free_grids
        total_state_coverage = np.count_nonzero(np.array(grid_occupancy_list).sum(axis=0)) / num_free_grids

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'external_reward_mean': np.mean(external_reward_mean_list),
            'current_std': np.mean(current_std_mean_list),
            'epoch_state_coverage': epoch_state_coverage,
            'total_state_coverage': total_state_coverage}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            episode_rewards = []
            episodes_state_coverage = []
            for i in range(cfg.eval_episodes):
                grids_visited_counts = np.zeros(env.maze_arr.shape)
                obs, done, ep_reward, t = env.reset(), False, 0, 0
                while not done:
                    hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
                    action, hidden, _ = agent.plan(obs, hidden, eval_mode=True, step=step, t0=t == 0)
                    obs, reward, done, _ = env.step(action.cpu().numpy())
                    append_data(eval_dataset, obs, action.cpu().numpy(), env.goal_locations[0],
                                done, env.sim.data)
                    free_grid_center = get_occupancy_idx(env.sim.data, free_grids_centers)
                    if free_grid_center is not None:
                        visited_grid = free_grid_center - 0.5
                        grids_visited_counts[int(visited_grid[0]), int(visited_grid[1])] += 1
                    ep_reward += reward
                    t += 1
                episode_rewards.append(ep_reward)
                eval_grid_occupancy_list.append(grids_visited_counts.reshape(-1))
                episodes_state_coverage.append(grids_visited_counts.reshape(-1))
            eval_total_state_coverage = np.count_nonzero(np.array(episodes_state_coverage).sum(axis=0)) / num_free_grids
            common_metrics['episode_reward'] = np.nanmean(episode_rewards)
            common_metrics['total_state_coverage'] = eval_total_state_coverage
            L.log(common_metrics, category='eval')
        if episode_idx % int(cfg.save_interval) == 0 and episode_idx <= 200:
            L.save_model(agent, episode_idx)

    dataset = h5py.File(work_dir / 'dataset.hdf5', 'w')
    npify(offline_dataset)
    for k in offline_dataset:
        dataset.create_dataset(k, data=offline_dataset[k], compression='gzip')
    grid_occupancy_list = np.array(grid_occupancy_list)
    with open(work_dir / 'grid_occupancy_list.pkl', 'wb') as f:
        pickle.dump(grid_occupancy_list, f)

    eval_offline_dataset = h5py.File(work_dir / 'eval_dataset.hdf5', 'w')
    npify(eval_dataset)
    for k in eval_dataset:
        eval_offline_dataset.create_dataset(k, data=eval_dataset[k], compression='gzip')
    eval_grid_occupancy_list = np.array(eval_grid_occupancy_list)
    with open(work_dir / 'eval_grid_occupancy_list.pkl', 'wb') as f:
        pickle.dump(eval_grid_occupancy_list, f)

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))