import gym
import os
import d4rl
import numpy as np
from pathlib import Path
from src.algorithm.ace_icem_similarity_drnn import TdICemSimDssm
from src.cfg import parse_cfg

__LOGS__, __CONFIG__ = 'logs', 'cfgs'


class ActRepeatWrapper(gym.Wrapper):
    def __init__(self, env, act_repeat):
        gym.Wrapper.__init__(self, env)
        self.act_repeat = act_repeat

    def step(self, action):
        if self.act_repeat == 1:
            obs, cum_reward, done, info = self.env.step(action)
        else:
            cum_reward = 0
            for _ in range(self.act_repeat):
                obs, reward, done, info = self.env.step(action)
                cum_reward += reward
                if done:
                    break
        return obs, cum_reward, done, info


def make_maze_env(cfg):
    env_id = cfg.task
    env = gym.make(env_id)
    env = ActRepeatWrapper(env, cfg.action_repeat)
    cfg.obs_shape = env.observation_space.shape
    cfg.buffer_shape = cfg.obs_shape
    cfg.action_shape = env.action_space.shape
    cfg.action_dim = env.action_space.shape[0]
    return env


def get_occupancy_idx(sim_data, free_grid_centers):
    qpos = sim_data.qpos.ravel()
    for free_grid_center in free_grid_centers:
        if np.allclose(qpos, free_grid_center, atol=0.5, rtol=0.0):
            return free_grid_center


def eval_policy(cfg):
    env = make_maze_env(cfg)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    agent = TdICemSimDssm(cfg)
    fp = os.path.join(work_dir, cfg.model_path)
    agent.load(fp)
    print('Loaded model from {}'.format(fp))
    episode_rewards = []
    episode_occupancy =[]
    num_rollouts = 10

    num_free_grids = len(env.empty_and_goal_locations)
    free_grids_centers = np.array(env.empty_and_goal_locations) - 0.5

    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        grids_visited_counts = np.zeros(env.maze_arr.shape)
        step_count, r_sum = 0, 0
        while not done:
            hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, _, _ = agent.plan(obs, hidden, eval_mode=True, step=0, t0=step_count == 0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            free_grid_center = get_occupancy_idx(env.sim.data, free_grids_centers)
            if free_grid_center is not None:
                visited_grid = free_grid_center - 0.5
                grids_visited_counts[int(visited_grid[0]), int(visited_grid[1])] += 1
            env.render()
            r_sum += reward
            step_count += 1
        episode_rewards.append(r_sum)
        num_visited_grids = np.count_nonzero(grids_visited_counts)
        epoch_state_coverage = num_visited_grids / num_free_grids
        episode_occupancy.append(epoch_state_coverage)
    env.close()
    print('Average reward: {}'.format(sum(episode_rewards) / len(episode_rewards)),
          'Average occupancy {}'.format(sum(episode_occupancy) / num_rollouts))


if __name__ == '__main__':
    eval_policy(parse_cfg(Path().cwd() / __CONFIG__))
