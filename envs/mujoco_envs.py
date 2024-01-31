import gym
import os
from pathlib import Path
import numpy as np
import robohive
import time
from robohive.utils import tensor_utils
from gym.wrappers import RecordEpisodeStatistics, NormalizeReward
from mjrl.utils.gym_env import GymEnv
from src.algorithm.ace_icem_similarity_reset import AceReset
from src.cfg import parse_cfg


__LOGS__, __CONFIG__ = 'logs', 'cfgs'


class ActRepeatWrapper(gym.Wrapper):
    """
    Wrapper for mujoco environments to repeat actions.
    """

    def __init__(self, env, act_repeat):
        gym.Wrapper.__init__(self, env)
        self.act_repeat = act_repeat

    def step(self, action):
        if self.act_repeat == 1:
            obs, cum_reward, done, info = self.env.step(action)
            env_infos = [info]
        else:
            cum_reward = 0
            env_infos = []
            for _ in range(self.act_repeat):
                obs, reward, done, info = self.env.step(action)
                cum_reward += reward
                env_infos.append(info)
                if done:
                    break
        return obs, cum_reward, done, env_infos


class RoboHiveWrapper(gym.Env):
    def __init__(self, env, act_repeat):
        super().__init__()
        self.env = env
        self.act_repeat = act_repeat

    # def _get_obs(self):
    #     obs_dict = self.env.get_env_infos()['obs_dict']
    #     palm_pos = obs_dict['palm_pos']
    #     hand_jnt = obs_dict['hand_jnt']
    #     hand_vel = obs_dict['hand_vel']
    #     obj_pos = obs_dict['obj_pos']
    #     obj_vel = obs_dict['obj_vel']
    #     obj_rot = obs_dict['obj_rot']
    #     tool_pos = obs_dict['tool_pos']
    #     target_pos = obs_dict['target_pos']
    #     goal_pos = obs_dict['goal_pos']
    #     # nail_impact = obs_dict['nail_impact']
    #     # construct obs for ace
    #     goal_error = target_pos - goal_pos
    #     target_goal_dist = np.linalg.norm(goal_error)
    #     nail_impact = target_goal_dist
    #     # if nail_impact is not
    #     return np.concatenate((palm_pos, hand_jnt, hand_vel,
    #                            obj_pos, obj_rot, obj_vel,
    #                            goal_pos, nail_impact.reshape(1, ),
    #                            tool_pos, target_pos))

    def _get_obs(self):
        obs_dict = self.env.get_env_infos()['obs_dict']
        palm_pos = obs_dict['palm_pos']
        hand_jnt = obs_dict['hand_jnt']
        handle_pos = obs_dict['handle_pos']
        latch_pos = obs_dict['latch_pos']
        reach_err = obs_dict['reach_err']  # palm to handle relative pos
        door_pos = obs_dict['door_pos']  # door pos > 1.35 means door open (scaler)
        door_pos_delta = door_pos - 1.35
        latch_pos_delta = latch_pos - 1.80

        return np.concatenate((palm_pos, hand_jnt,
                               door_pos_delta.reshape(1, ), latch_pos_delta.reshape(1, ), handle_pos, reach_err))

    def reset(self):
        self.env.reset()
        return self._get_obs()

    def step(self, action):
        reward = 0
        env_infos = []
        for _ in range(self.act_repeat):
            obs, r, done, info = self.env.step(action)
            reward += r
            env_infos.append(info)
            if done:
                break
        return self._get_obs(), reward, done, env_infos

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


def make_robohive_env(cfg):
    env_id = cfg.task
    words = env_id.split('-')
    if words[0] in ['Franka', 'Trifinger']:
        task = ''.join(words[:2])
        env_id = task + '-' + words[-1]
    env = gym.make(env_id)
    # env = ActRepeatWrapper(env, cfg.action_repeat)
    env = RoboHiveWrapper(env, cfg.action_repeat)

    obs = env.reset()
    cfg.buffer_shape = (obs.shape[0],)
    cfg.obs_shape = (obs.shape[0],)
    cfg.action_shape = (env.action_space.shape[0],)
    cfg.action_dim = env.action_space.shape[0]
    return env


class ICEMPlanner:
    def __init__(self, cfg, work_dir):
        self.cfg = cfg
        self.agent = AceReset(cfg)
        fp = os.path.join(work_dir, cfg.model_path)
        self.agent.load(fp)
        print('have loaded the model from {}'.format(fp))

    def get_action(self, obs, env_step):
        hidden = self.agent.model.init_hidden_state(batch_size=1, device='cuda')
        action, _, _ = self.agent.plan(obs, hidden, eval_mode=True, step=0, t0=env_step == 0)
        return action.cpu().numpy()

    # def get_action(self, obs, env_step):
    #     obs = torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0)
    #     action = self.agent.model.pi(self.agent.model.h(obs))
    #     action = action.squeeze().detach().cpu().numpy()
    #     return action


def get_paths(env, planner, work_dir):
    paths = env.examine_policy_new(
        policy=planner,
        horizon=env.spec.max_episode_steps,
        num_episodes=10,
        frame_size=(640, 480),
        mode='exploration',  # fake exploration for easy test
        output_dir=work_dir / 'videos',
        filename='icem_planner',
        camera_name=None,
        render='onscreen',
    )
    return paths


def eval_policy(cfg):
    env = make_robohive_env(cfg)
    env.reset()
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    planner = ICEMPlanner(cfg, work_dir)
    episode_rewards, paths = [], []
    for i in range(10):
        env_infos = []
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        print('current epoch is : ', i)
        while not done:
            time.sleep(0.05)
            env.mj_render()
            action = planner.get_action(obs, t)
            obs, reward, done, env_info = env.step(action)
            obs_dict = env.get_obs_dict(env.sim)
            print(obs_dict['door_pos'], obs_dict['latch_pos'], obs_dict['door_open'])
            env_infos.extend(env_info)
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
        path = dict(env_infos=tensor_utils.stack_tensor_dict_list(env_infos))
        paths.append(path)
        print('cum episode reward is : ', ep_reward)
    success_percentage = env.env.evaluate_success(paths, successful_steps=5)
    print(f'Average success over rollouts: {success_percentage}%')


if __name__ == '__main__':
    eval_policy(parse_cfg(Path().cwd() / __CONFIG__))
    # import robohive
    # env = gym.make('FrankaPickPlaceRandom-v0')
    # env.reset()
    # cum_reward = 0
    # counts = 0
    # for t in range(1000):
    #     env.mj_render()
    #     o, r, d, info = env.step(env.action_space.sample())
    #     # print(env.sim.data.site_xpos[env.object_sid])
    #     cum_reward += r
    #     counts += 1
    #     time.sleep(0.1)
    #     # print(r)
    #     if d:
    #         print('episode_done', info["solved"], info["done"], cum_reward, counts)
    #         env.reset()
    #         d = False
    #         cum_reward = 0
    #         counts = 0
