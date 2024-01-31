import gym
import os
from pathlib import Path
import random
import numpy as np
import time
from gym.wrappers.time_limit import TimeLimit
from src.algorithm.ace_icem_similarity_drnn import AceDRNN
from src.algorithm.helper import symlog_np
from src.cfg import parse_cfg
from scipy.spatial.transform import Rotation as R
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


__LOGS__, __CONFIG__ = 'logs', 'cfgs'


class MetaWorldWrapper(gym.Env):
    """
    Wrapper for metaworld environments.
    """

    def __init__(self, env, act_repeat, feature_dim):
        super().__init__()
        self.env = env
        self.act_repeat = act_repeat
        self.feature = feature_dim
        self._state_obs = None

    # def _get_obs(self):
    #     obs = self._state_obs.copy()
    #     hand_pos = obs[:3]
    #     gripper_apart = obs[3]
    #     handle_pos = obs[4:7]
    #     hammer_head = handle_pos + np.array([.16, .06, .0])
    #     obj_quat = obs[7:11]
    #     r = R.from_quat(obj_quat)
    #     obj_euler = r.as_euler('xyz', degrees=False)
    #     obj_vel = self.env.data.qvel.flat.copy()[9:15]
    #
    #     nail_impact = self.env.data.get_joint_qpos('NailSlideJoint')
    #     obj_error = handle_pos - hand_pos
    #
    #     pos_goal = obs[-3:]
    #     goal_error = pos_goal - hammer_head
    #
    #     return np.concatenate((
    #         hand_pos,
    #         [gripper_apart, ],
    #         obj_error,
    #         obj_euler,
    #         obj_vel,
    #         goal_error,
    #         [nail_impact, ],
    #         # handle_pos,
    #         # pos_goal,
    #     ))

    def _get_obs(self):
        obs = self._state_obs.copy()
        hand_pos = obs[:3]
        gripper_apart = obs[3]
        obj_pos = obs[4:7]
        obj_quat = obs[7:11]
        r = R.from_quat(obj_quat)
        obj_euler = r.as_euler('xyz', degrees=False)
        pos_goal = obs[-3:]
        obj_vel = self.env.data.qvel.flat.copy()[0:3]
        obj_error = obj_pos - hand_pos

        # obj_center = self.env.get_body_com('RoundNut')
        obj_center = obj_pos
        goal_error = pos_goal - obj_center

        return np.concatenate((
            hand_pos,
            [gripper_apart, ],
            obj_error,
            obj_euler,
            obj_vel,
            goal_error,

            obj_center,
            pos_goal,
        ))

    def reset(self, **kwargs):
        self.env.reset()
        obs = self.env.step(np.zeros_like(self.env.action_space.sample()))[0].astype(np.float32)
        self._state_obs = obs
        return self._get_obs()

    def step(self, action):
        reward = 0
        env_infos = []
        for _ in range(self.act_repeat):
            obs, r, done, info = self.env.step(action)
            reward += 0.1 * info['grasp_success'] + 0.4 * info['success']
            env_infos.append(info)
            if done:
                break
        obs = obs.astype(np.float32)
        self._state_obs = obs
        return self._get_obs(), reward, done, env_infos

    def set_task(self, tasks):
        self.env.set_task(tasks)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def mj_render(self):
        self.env.mj_render()


def make_meta_world_env(cfg, feature_dim=7):
    # set up a single environment form the metaworld benchmark
    env_id = cfg.task
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id + '-goal-observable'](seed=cfg.seed)
    env._freeze_rand_vec = True
    env = TimeLimit(env, max_episode_steps=100)
    env = MetaWorldWrapper(env, cfg.action_repeat, feature_dim)

    obs = env.reset()
    cfg.buffer_shape = (obs.shape[0],)
    cfg.obs_shape = (obs.shape[0] - cfg.goal_dim,)
    cfg.action_shape = (env.action_space.shape[0],)
    cfg.action_dim = env.action_space.shape[0]

    return env


def eval_policy(cfg):
    env = make_meta_world_env(cfg)
    env.reset()
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    agent = AceDRNN(cfg)
    fp = os.path.join(work_dir, cfg.model_path)
    agent.load(fp)
    print('have loaded the model from {}'.format(fp))

    episode_rewards = []
    successful_episodes = 0
    for i in range(10):
        success_count = 0
        # env.set_task(random.choice(tasks))
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        while not done:
            time.sleep(0.05)
            env.mj_render()
            hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            s, _ = np.split(obs, [cfg.obs_shape[0]], axis=-1)
            action, hidden, _ = agent.plan(s, hidden, eval_mode=True, step=0, t0=t == 0)
            obs, reward, done, infos = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
            for info in infos:
                success_count += info['success']  # 10 for default in meta world.
        print('episode {}, reward {}, success_count {}'.format(i, ep_reward, success_count))
        episode_rewards.append(ep_reward)
        if success_count >= 10:
            successful_episodes += 1
    success_percentage = (successful_episodes / 10) * 100
    print('success_percentage is: ', success_percentage,
          'episode_rewards is: ', np.nanmean(episode_rewards))


if __name__ == "__main__":
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    eval_policy(cfg)
