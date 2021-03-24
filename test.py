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
        self.evader_ob_rms = RunningMeanStd(shape=self.env.evader_observation_space.shape) if ob else None
        self.pursuer1_ob_rms = RunningMeanStd(shape=self.env.pursuer_observation_space.shape) if ob else None
        self.pursuer2_ob_rms = RunningMeanStd(shape=self.env.pursuer_observation_space.shape) if ob else None
        self.clipob = clipob
        self.epsilon = epsilon
        self.times = 0
        self.evader_obs = []
        self.pursuer1_obs = []
        self.pursuer2_obs = []
        self.evader_ob = []
        self.pursuer1_ob = []
        self.pursuer2_ob = []

    def step(self, action):
        obs, rews, done, infos = self.env.step(action)
        self.times += 1
        self.evader_obs.append(obs[0])
        self.evader_ob.append(obs[0])
        self.pursuer1_obs.append(obs[1])
        self.pursuer1_ob.append(obs[1])
        self.pursuer2_obs.append(obs[2])
        self.pursuer2_ob.append(obs[2])
        if self.times% 50== 0:
        	self.evader_ob_rms.update(np.array(self.evader_ob))
        	self.pursuer1_ob_rms.update(np.array(self.pursuer1_ob))
        	self.pursuer2_ob_rms.update(np.array(self.pursuer2_ob))
        	print(self.evader_ob_rms.mean, self.pursuer1_ob_rms.mean, self.pursuer2_ob_rms.mean)        	
        	self.evader_ob = []
        	self.pursuer1_ob = []        	
        	self.pursuer2_ob = []
        evader_ob = self.evader_obfilt(obs[0])
        pursuer_ob_1 = self.pursuer1_obfilt(obs[1])
        pursuer_ob_2 = self.pursuer2_obfilt(obs[2])
        obs = np.array([evader_ob, pursuer_ob_1, pursuer_ob_2])
        return obs, rews, done, infos

    def evader_obfilt(self, obs):
        if self.evader_ob_rms:
            # self.evader_ob_rms.update(obs)
            obs = np.clip((obs - self.evader_ob_rms.mean) / np.sqrt(self.evader_ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

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
        evader_ob = self.evader_obfilt(obs[0])
        pursuer_ob_1 = self.pursuer1_obfilt(obs[1])
        pursuer_ob_2 = self.pursuer2_obfilt(obs[2])
        obs = np.array([evader_ob, pursuer_ob_1, pursuer_ob_2])
        return obs
        
    def render(self):
        self.env.render()


# ------------------------------------- #

env = gym.make('PursuitWorld-v1')
normal_env = Normalize(env)

#第一步不用agent，采用随机策略进行对比
normal_env.reset() #初始化环境
random_episodes = 0
reward_sum = 0
train_step = 0
sum_obs = np.array([[0.0 for i in range(17)] for j in range(3)])
mean_obs = np.array([[0.0 for i in range(17)] for j in range(3)])
var_obs = np.array([[0.0 for i in range(17)] for j in range(3)])
evader_obs = []
pursuer1_obs = []
pursuer2_obs = []
while random_episodes < 1:
    normal_env.render()
    action = np.array([int(np.random.uniform(0, 4))/2*np.pi, int(np.random.uniform(0, 4))/2*np.pi, int(np.random.uniform(0, 4))/2*np.pi])
    obsevation, reward, done_, _ = normal_env.step(action)
    evader_obs.append(obsevation[0])
    pursuer1_obs.append(obsevation[1])
    pursuer2_obs.append(obsevation[2])
    reward_sum += reward
    train_step += 1
    if train_step == 10000:
    	evader_obs = np.array(evader_obs)
    	pursuer1_obs = np.array(pursuer1_obs)
    	pursuer2_obs = np.array(pursuer2_obs)
    	
    	last_evader_obs = np.array(normal_env.evader_obs)
    	last_pursuer1_obs = np.array(normal_env.pursuer1_obs)
    	last_pursuer2_obs = np.array(normal_env.pursuer2_obs)
    	
    	print(evader_obs.shape, last_evader_obs.shape)
    	
    	last_evader_mean = np.mean(last_evader_obs, axis=0)
    	last_pursuer1_mean = np.mean(last_pursuer1_obs, axis=0)
    	last_pursuer2_mean = np.mean(last_pursuer2_obs, axis=0)
    	print(last_evader_mean, last_pursuer1_mean, last_pursuer2_mean)
    	
    	last_evader_var = np.var(last_evader_obs, axis=0)
    	last_pursuer1_var = np.var(last_pursuer1_obs, axis=0)
    	last_pursuer2_var = np.var(last_pursuer2_obs, axis=0)
    	
    	last_evader_obs = np.clip((last_evader_obs - last_evader_mean) / np.sqrt(last_evader_var + normal_env.epsilon), -normal_env.clipob, normal_env.clipob)
    	last_pursuer1_obs = np.clip((last_pursuer1_obs - last_pursuer1_mean) / np.sqrt(last_pursuer1_var + normal_env.epsilon), -normal_env.clipob, normal_env.clipob)
    	last_pursuer2_obs = np.clip((last_pursuer2_obs - last_pursuer2_mean) / np.sqrt(last_pursuer2_var + normal_env.epsilon), -normal_env.clipob, normal_env.clipob)
    	
    	print(last_evader_var.shape, last_evader_mean.shape)
    	print(last_pursuer2_obs.shape)
    	print(evader_obs-last_evader_obs)
    	
    	random_episodes += 1
    	# print("Reward for this episodes was:", reward_sum)
    	# print("train_step:", train_step)
    	reward_sum = 0  #重置reward
    	train_step = 0
    	normal_env.reset()
    	obs = []
    	normal_env.obs = []
env.close()