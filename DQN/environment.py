"""

### NOTICE ###
DO NOT revise this file

"""
import gym
import ale_py
import numpy as np
from atari_wrapper import make_wrap_atari

class Environment(object):
    def __init__(self, env_name, args, atari_wrapper=False, test=False):
        if atari_wrapper:
            clip_rewards = not test
            self.env = make_wrap_atari(env_name, clip_rewards, args.do_render)
        else:
            self.env = gym.wrappers.RecordVideo(gym.make(env_name, render_mode="rgb_array"), video_folder="videos", episode_trigger=lambda x: True)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        '''
        observation, info = self.env.reset(seed=seed)

        return np.array(observation), info


    def step(self,action):
        '''
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
            reward: int
                if opponent wins, reward = +1 else -1
            done: bool
                whether reach the end of the episode?
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, truncated, info = self.env.step(action)

        return np.array(observation), reward, done, truncated, info


    def get_action_space(self):
        return self.action_space


    def get_observation_space(self):
        return self.observation_space


    def get_random_action(self):
        return self.action_space.sample()
