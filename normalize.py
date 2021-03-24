#!/usr/bin/env python3

import gym
import numpy as np

from pursuer_evasion import PursuitEvaderEnv

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        # print(batch_count)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
    

class Normalize():
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, env, ob=True, ret=True, clipob=10., epsilon=1e-8):
        self.env = env
        # self.evader_ob_rms = RunningMeanStd(shape=self.env.evader_observation_space.shape) if ob else None
        self.pursuer1_ob_rms = RunningMeanStd(shape=self.env.pursuer_observation_space.shape) if ob else None
        self.pursuer2_ob_rms = RunningMeanStd(shape=self.env.pursuer_observation_space.shape) if ob else None
        self.clipob = clipob
        self.epsilon = epsilon
        self.times = 0
        # self.evader_ob = []
        self.pursuer1_ob = []
        self.pursuer2_ob = []
        self.num = 50

    def step(self, action):
        obs, rews, done, infos = self.env.step(action)
        self.times += 1
        # self.evader_ob.append(obs[0])
        self.pursuer1_ob.append(obs[0])
        self.pursuer2_ob.append(obs[1])
        if self.times% self.num == 0:
        	# self.evader_ob_rms.update(np.array(self.evader_ob))
        	self.pursuer1_ob_rms.update(np.array(self.pursuer1_ob))
        	self.pursuer2_ob_rms.update(np.array(self.pursuer2_ob))
        	# self.evader_ob = []
        	self.pursuer1_ob = []        	
        	self.pursuer2_ob = []
        # evader_ob = self.evader_obfilt(obs[0])
        pursuer_ob_1 = self.pursuer1_obfilt(obs[0])
        pursuer_ob_2 = self.pursuer2_obfilt(obs[1])
        obs = np.array([pursuer_ob_1, pursuer_ob_2])
        return obs, rews, done, infos

    def pursuer1_obfilt(self, obs):
        if self.pursuer1_ob_rms:
            # self.pursuer1_ob_rms.update(obs)
            obs = np.clip((obs - self.pursuer1_ob_rms.mean) / np.sqrt(self.pursuer1_ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def pursuer2_obfilt(self, obs):
        if self.pursuer2_ob_rms:
            # self.pursuer2_ob_rms.update(obs)
            obs = np.clip((obs - self.pursuer2_ob_rms.mean) / np.sqrt(self.pursuer2_ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs


    def reset(self):
        obs = self.env.reset()
        # evader_ob = self.evader_obfilt(obs[0])
        pursuer_ob_1 = self.pursuer1_obfilt(obs[0])
        pursuer_ob_2 = self.pursuer2_obfilt(obs[1])
        obs = np.array([pursuer_ob_1, pursuer_ob_2])
        return obs
        
    def render(self):
        self.env.render()
