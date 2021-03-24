#!/usr/bin/env python3

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.linalg import solve
from matplotlib.patches import Circle
import numpy as np
import random
import time

# Evader is still

class PursuitEvaderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    
    def __init__(self):
    	self.kinematics_integrator = 'euler'
    	self.viewer = None
    	
    	# Pursuers Space
    	pursuer_high = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    	pursuer_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    	self.pursuer_action_space = spaces.Box(low = 0.0, high = 2*np.pi, shape=(1,))
    	self.pursuer_observation_space = spaces.Box(low = pursuer_low, high = pursuer_high)
    	
    	self.tau = 0.5
    	self.e_x = 0.0
    	self.e_y = 0.0
    	self.alpha = 0.2
    	
    	self.state = None
    	self.seed()
    	self.steps_beyond_done = None
    	self.steps = 0
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
    	# Get state data
    	# Get pursuers' and evader's params
    	p1_x, p1_y, p2_x, p2_y, e_x, e_y = self.state[0][12:]
    	for i in range(2):
    		for j in range(6):
    			self.state[i][j] = self.state[i][j+6]
    			self.state[i][j+6] = self.state[i][j+12]
    	
    	# Clip
    	np.clip(action, 0.0, 2*np.pi)
    	cost = 0.0
    	
    	# Get action
    	# print(self.state)
    	pursuer_action = np.array([action[0][0], action[1][0]])
    	
    	# Pursuer running
    	# Pursuer No.1
    	p1 = np.array([p1_x, p1_y])
    	dp1_x = 0.8*self.tau*np.cos(pursuer_action[0])
    	dp1_y = 0.8*self.tau*np.sin(pursuer_action[0])
    	# Pursuer N0.2
    	p2 = np.array([p2_x, p2_y])
    	dp2_x = 0.8*self.tau*np.cos(pursuer_action[1])
    	dp2_y = 0.8*self.tau*np.sin(pursuer_action[1])
    	# Evader
    	e = np.array([self.e_x, self.e_y])
    	de_x = 0.0
    	de_y = 0.0
    	
    	# Get Last Distance
    	last_distance = []
    	last_distance.append(np.sqrt(np.sum(np.square(p1-e))))
    	last_distance.append(np.sqrt(np.sum(np.square(p2-e))))
    	
    	# Collision the wall
    	# Pursuer No.1
    	if p1[0] + dp1_x < 0:
    		# # print("Pursuer No.1 collision the wall")
    		dp1_x = -p1[0]
    		cost = -0.25
    		# time.sleep(1)
    	elif p1[0] + dp1_x >100:
    		# # print("Pursuer No.1 collision the wall")
    		dp1_x= 100 - p1[0]
    		cost = -0.25
    		# time.sleep(1)
    	if p1[1] + dp1_y < 0:
    		# # print("Pursuer No.1 collision the wall")
    		dp1_y = -p1[1]
    		cost = -0.25
    		# time.sleep(1)
    	elif p1[1] + dp1_y >100:
    		# # print("Pursuer No.1 collision the wall")
    		dp1_y = 100 - p1[1]
    		cost = -0.25
    		# time.sleep(1)
    	# Pursuer N0.2
    	if p2[0] + dp2_x < 0:
    		# # print("Pursuer No.2 collision the wall")
    		dp2_x = -p2[0]
    		cost = -0.25
    		# time.sleep(1)
    	elif p2[0] + dp2_x >100:
    		# # print("Pursuer No.2 collision the wall")
    		dp2_x= 100 - p2[0]
    		cost = -0.25
    		# time.sleep(1)
    	if p2[1] + dp2_y < 0:
    		# # print("Pursuer No.2 collision the wall")
    		dp2_y = -p2[1]
    		cost = -0.25
    		# time.sleep(1)
    	elif p2[1] + dp2_y >100:
    		# # print("Pursuer No.2 collision the wall")
    		dp2_y = 100 - p2[1]
    		cost = -0.25
    	
    	p1[0] = p1[0] + dp1_x
    	p1[1] = p1[1] + dp1_y
    	p2[0] = p2[0] + dp2_x
    	p2[1] = p2[1] + dp2_y
    	e[0] = e[0] + de_x
    	e[1] = e[1] + de_y
    	self.e_x = e[0]
    	self.e_y = e[1]
    	
    	# Compute the velocity of agents
    	agent_v = []
    	agent_v.append([dp1_x, dp1_y])
    	agent_v.append([dp2_x, dp2_y])
    	agent_v = np.array(agent_v)
    	agent_v_angle = np.array([0.0, 0.0, 0.0])
    	
    	for i in range(0, 2):
    		if agent_v[i][0] > 0 and agent_v[i][1] >= 0:
    			agent_v_angle[i] = math.atan(agent_v[i][1]/agent_v[i][0])
    		elif agent_v[i][0] > 0 and agent_v[i][1] < 0:
    			agent_v_angle[i] = math.atan(agent_v[i][1]/agent_v[i][0]) + 2*np.pi
    		elif agent_v[i][0] < 0 and agent_v[i][1] >= 0:
    			agent_v_angle[i] = math.atan(agent_v[i][1]/agent_v[i][0]) + np.pi
    		elif agent_v[i][0] < 0 and agent_v[i][1] < 0:
    			agent_v_angle[i] = math.atan(agent_v[i][1]/agent_v[i][0]) + np.pi
    		elif agent_v[i][0] == 0:
    			if agent_v[i][1] > 0:
    				agent_v_angle[i] = 1/2*np.pi
    			else:
    				agent_v_angle[i] = 3/2*np.pi
    	
    	self.steps_beyond_done = None
    	
    	# Get New Distance
    	New_distance = []
    	New_distance.append(np.sqrt(np.sum(np.square(p1-e))))
    	New_distance.append(np.sqrt(np.sum(np.square(p2-e))))
    	# print(New_distance)
    	
    	self.state[0][12:] = p1[0], p1[1], p2[0], p2[1], e[0], e[1]
    	self.state[1][12:] = p2[0], p2[1], p1[0], p1[1], e[0], e[1]
    	
    	success= (np.sqrt(np.sum(np.square(p1-e))) <= 2 or np.sqrt(np.sum(np.square(p2-e))) <= 2)
    	done = success or self.steps >= 599    	
    	done = bool(done)
    	
    	distance_difference = [New_distance[0]-last_distance[0], New_distance[1]-last_distance[1]]
    	# print(distance_difference)
    	
    	if done == 0:
    		self.steps += 1
    		exploration_pursuer_reward_1 = (-1)*distance_difference[0]*0.5
    		exploration_pursuer_reward_2 = (-1)*distance_difference[1]*0.5
    		goal_pursuer_reward_1 = 0.0
    		goal_pursuer_reward_2 = 0.0
    	elif self.steps_beyond_done is None:
    		self.steps_beyond_done = 0
    		exploration_pursuer_reward_1 = (-1)*distance_difference[0]*0.5
    		exploration_pursuer_reward_2 = (-1)*distance_difference[1]*0.5
    		if success:
    			goal_pursuer_reward_1 = 800.0
    			goal_pursuer_reward_2 = 800.0
    		else:
    			goal_pursuer_reward_1 = -800.0
    			goal_pursuer_reward_2 = -800.0
    	else:
    		if self.steps_beyond_done == 0:
    			logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior.")
    		self.steps_beyond_done = self.steps_beyond_done + 1
    		exploration_pursuer_reward_1 = (-1)*distance_difference[0]*0.5
    		exploration_pursuer_reward_2 = (-1)*distance_difference[1]*0.5
    		if success:
    			goal_pursuer_reward_1 = 800.0
    			goal_pursuer_reward_2 = 800.0
    		else:
    			goal_pursuer_reward_1 = -800.0
    			goal_pursuer_reward_2 = -800.0
    	pursuer_reward_1 = self.alpha * exploration_pursuer_reward_1 + (1-self.alpha) * goal_pursuer_reward_1 + cost
    	pursuer_reward_2 = self.alpha * exploration_pursuer_reward_2 + (1-self.alpha) * goal_pursuer_reward_2 + cost
    	reward = np.array([pursuer_reward_1, pursuer_reward_2])
    	
    	info = {'exploration_pursuer_reward_1':exploration_pursuer_reward_1, 'exploration_pursuer_reward_2':exploration_pursuer_reward_2, 'goal_pursuer_reward_1':goal_pursuer_reward_1, 'goal_pursuer_reward_2':goal_pursuer_reward_2}
    	return self.state, reward, done, info
    	
    def reset(self):
    	# Initial the high and low
    	pursuer_state = self.np_random.uniform(low=pursuer_low, high=pursuer_high)
    	pursuer_high = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    	pursuer_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    	pursuer_state = self.np_random.uniform(low=pursuer_low, high=pursuer_high)
    	
    	self.steps = 0
    	
    	# Initial the reset pose
    	# evader_state[4] = random.uniform(0.0, 100.0)
    	region = random.randint(0,1)
    	if region == 0:
    		self.e_x = random.uniform(0.0, 5.0) + 95.0 * random.randint(0,1)
    	else:
    		self.e_x = random.uniform(5.0+1e-7, 95.0-1e-7)
    	if self.e_x <= 5.0 or self.e_x >= 95.0:
    		self.e_y = random.uniform(0.0, 100.0)
    	else:
    		self.e_y = random.uniform(0.0, 5.0) + 95.0 * random.randint(0,1)
    	
    	pursuer_state[0][0] = random.uniform(40.0, 60.0)
    	pursuer_state[0][1] = random.uniform(40.0, 60.0)
    	pursuer_state[1][0] = random.uniform(40.0, 60.0)
    	pursuer_state[1][1] = random.uniform(40.0, 60.0)
    	
    	pursuer_state[0][2] = pursuer_state[1][0]
    	pursuer_state[0][3] = pursuer_state[1][1]
    	pursuer_state[1][2] = pursuer_state[0][0]
    	pursuer_state[1][3] = pursuer_state[0][1]
    	
    	pursuer_state[0][4] = self.e_x
    	pursuer_state[0][5] = self.e_y
    	pursuer_state[1][4] = self.e_x
    	pursuer_state[1][5] = self.e_y
    	
    	
    	states = [[0] * (18) for _ in range(2)]
    	for i in range(2):
    		for j in range(6):
    			states[i][j] = pursuer_state[i][j]
    			states[i][j+6] = pursuer_state[i][j]
    			states[i][j+12] = pursuer_state[i][j]
    	
    	self.state = states
    	self.steps_beyond_done = None
    	
    	return np.array(self.state)
        
    def render(self, mode='human', close=False):
    	from gym.envs.classic_control import rendering
    	screen_width = 410
    	screen_height = 410
    	if self.viewer is None:
    		self.viewer = rendering.Viewer(screen_width, screen_height)
    		
    		self.line1 = rendering.Line((105, 105), (105, 305))
    		self.line2 = rendering.Line((105, 105), (305, 105))
    		self.line3 = rendering.Line((305, 105), (305, 305))
    		self.line4 = rendering.Line((305, 305), (105, 305))
    		
    		self.line1.set_color(0, 0, 0)
    		self.line2.set_color(0, 0, 0)
    		self.line3.set_color(0, 0, 0)
    		self.line4.set_color(0, 0, 0)
    		
    		self.obstacle1 = rendering.make_circle(30, 120)
    		self.obs1trans = rendering.Transform(translation=(265, 265))
    		self.obstacle1.add_attr(self.obs1trans)
    		self.viewer.add_geom(self.obstacle1)
    		
    		self.obstacle2 = rendering.make_circle(40, 160)
    		self.obs2trans = rendering.Transform(translation=(155, 155))
    		self.obstacle2.add_attr(self.obs2trans)
    		self.viewer.add_geom(self.obstacle2)
    		
    		self.pursuer1 = rendering.make_circle(1)
    		self.p1trans = rendering.Transform()
    		self.pursuer1.add_attr(self.p1trans)
    		self.pursuer1.set_color(0, 1, 0)
    		
    		self.pursuer1_capture = rendering.make_circle(3, 30, filled=False)
    		self.p1ctrans = rendering.Transform()
    		self.pursuer1_capture.add_attr(self.p1ctrans)
    		self.pursuer1_capture.set_color(0, 1, 0)
    		
    		self.pursuer2 = rendering.make_circle(1)
    		self.p2trans = rendering.Transform()
    		self.pursuer2.add_attr(self.p2trans)
    		self.pursuer2.set_color(0, 1, 0)
    		
    		self.pursuer2_capture = rendering.make_circle(3, 30, filled=False)
    		self.p2ctrans = rendering.Transform()
    		self.pursuer2_capture.add_attr(self.p2ctrans)
    		self.pursuer2_capture.set_color(0, 1, 0)
    		
    		self.evader = rendering.make_circle(2)
    		self.etrans = rendering.Transform()
    		self.evader.add_attr(self.etrans)
    		self.evader.set_color(0, 0, 1)
    		
    		self.viewer.add_geom(self.line1)
    		self.viewer.add_geom(self.line2)
    		self.viewer.add_geom(self.line3)
    		self.viewer.add_geom(self.line4)
    		self.viewer.add_geom(self.pursuer1)
    		self.viewer.add_geom(self.pursuer1_capture)
    		self.viewer.add_geom(self.pursuer2)
    		self.viewer.add_geom(self.pursuer2_capture)
    		self.viewer.add_geom(self.evader)
    		
    	if self.state is None:
    		return None
    	
    	self.p1trans.set_translation(self.state[0][0]*2+105, self.state[0][1]*2+105)
    	self.p1ctrans.set_translation(self.state[0][0]*2+105, self.state[0][1]*2+105)
    	self.p2trans.set_translation(self.state[1][0]*2+105, self.state[1][1]*2+105)
    	self.p2ctrans.set_translation(self.state[1][0]*2+105, self.state[1][1]*2+105)
    	self.etrans.set_translation(self.e_x*2+105, self.e_y*2+105)
    	
    	return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            