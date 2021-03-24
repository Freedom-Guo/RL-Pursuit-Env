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

# Evader randomly
# speed: == pursuer speed

class PursuerLevelFour(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    
    def __init__(self):
    	self.kinematics_integrator = 'euler'
    	self.viewer = None
    	
    	# Pursuers Space
    	pursuer_high= np.array([100.0, 100.0, 2*np.pi, 100.0*(2**0.5), 2*np.pi, 100.0*(2**0.5), 2*np.pi, 100.0, 100.0, 100.0, 100.0, (80**2+80**2)**0.5, 2*np.pi, 15, (75**2+75**2)**0.5, 2*np.pi, 20])
    	pursuer_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20])
    	self.pursuer_action_space = spaces.Box(low = 0.0, high = 2*np.pi, shape=(1,))
    	self.pursuer_observation_space = spaces.Box(low = pursuer_low, high = pursuer_high)
    	
    	self.tau = 0.5
    	self.e_x = 0.0
    	self.e_y = 0.0
    	self.alpha = 1
    	
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
    	p1_x, p1_y, p1_v, p1_p2_r, p1_p2_angle, p1_e_r, p1_e_angle, p1_l1, p1_l2, p1_l3, p1_l4, p1_ob1_r, p1_ob1_angle, ob1_r, p1_ob2_r, p1_ob2_angle, ob2_r = self.state[0]
    	p2_x, p2_y, p2_v, p2_p1_r, p2_p1_angle, p2_e_r, p2_e_angle, p2_l1, p2_l2, p2_l3, p2_l4, p2_ob1_r, p2_ob1_angle, ob1_r, p2_ob2_r, p2_ob2_angle, ob2_r = self.state[1]
    	
    	# Clip
    	np.clip(action, 0.0, 2*np.pi)
    	cost = 0.0
    	
    	# Get action
    	pursuer_action = np.array([action[0][0], action[1][0]])
    	evader_action = random.uniform(0.0, 2*np.pi)
    	
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
    	de_x = 0.8*self.tau*np.cos(evader_action)
    	de_y = 0.8*self.tau*np.sin(evader_action)
    	
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
    		# time.sleep(1)
    	# Evader
    	if e[0] + de_x < 0:
    		# # print("Evader collision the wall")
    		de_x = -e[0]
    		# time.sleep(1)
    	elif e[0] + de_x >100:
    		# # print("Evader collision the wall")
    		de_x= 100 - e[0]
    		# time.sleep(1)
    	if e[1] + de_y < 0:
    		# # print("Evader collision the wall")
    		de_y = -p2[1]
    		# time.sleep(1)
    	elif e[1] + de_y >100:
    		# # print("Evader collision the wall")
    		de_y = 100 - e[1]
    	
    	# Collision the obstacle
    	ob1 = np.array([80, 80])
    	ob2 = np.array([25, 25])
    	# Pursuer No.1
    	p_new = np.array([p1[0]+dp1_x, p1[1]+dp1_y])
    	dis = np.sqrt(np.sum(np.square(p_new - ob1)))
    	if dis < ob1_r:
    		circle = [ob1[0], ob1[1], ob1_r]
    		temp = [0, 0]
    		temp[0], temp[1] = LineIntersectCircle(circle, p1, p_new)
    		dp1_x, dp1_y = get_pose(ob1, temp, p_new, p1)
    		cost = -0.25
    	dis = np.sqrt(np.sum(np.square(p_new - ob2)))
    	if dis < ob2_r:
    		circle = [ob2[0], ob2[1], ob2_r]
    		temp = [0, 0]
    		temp[0], temp[1] = LineIntersectCircle(circle, p1, p_new)
    		dp1_x, dp1_y = get_pose(ob2, temp, p_new, p1)
    		cost = -0.25
    	# Pursuer No.2
    	p_new = np.array([p2[0]+dp2_x, p2[1]+dp2_y])
    	dis = np.sqrt(np.sum(np.square(p_new - ob1)))
    	if dis < ob1_r:
    		circle = [ob1[0], ob1[1], ob1_r]
    		temp = [0, 0]
    		temp[0], temp[1] = LineIntersectCircle(circle, p2, p_new)
    		dp2_x, dp2_y = get_pose(ob1, temp, p_new, p2)
    		cost = -0.25
    	dis = np.sqrt(np.sum(np.square(p_new - ob2)))
    	if dis < ob2_r:
    		circle = [ob2[0], ob2[1], ob2_r]
    		temp = [0, 0]
    		temp[0], temp[1] = LineIntersectCircle(circle, p2, p_new)
    		dp2_x, dp2_y = get_pose(ob2, temp, p_new, p2)
    		cost = -0.25
    	# Evader
    	p_new = np.array([e[0]+de_x, e[1]+de_y])
    	dis = np.sqrt(np.sum(np.square(p_new - ob1)))
    	# # print(dis)
    	if dis < ob1_r:
    		circle = [ob1[0], ob1[1], ob1_r]
    		temp = [0, 0]
    		temp[0], temp[1] = LineIntersectCircle(circle, e, p_new)
    		de_x, de_y = get_pose(ob1, temp, p_new, e)
    	dis  = np.sqrt(np.sum(np.square(p_new - ob2)))
    	# # print(dis)
    	if dis < ob2_r:
    		# de_x, de_y = 0, 0
    		# print("Evader collision obstacle 2")
    		circle = [ob2[0], ob2[1], ob2_r]
    		temp = [0, 0]
    		temp[0], temp[1] = LineIntersectCircle(circle, e, p_new)
    		de_x, de_y = get_pose(ob2, temp, p_new, e)
    	
    	p1[0] = p1[0] + dp1_x
    	p1[1] = p1[1] + dp1_y
    	p2[0] = p2[0] + dp2_x
    	p2[1] = p2[1] + dp2_y
    	e[0] = e[0] + de_x
    	e[1] = e[1] + de_y
    	self.e_x = e[0]
    	self.e_y = e[1]
    	# print(self.e_x, self.e_y)
    	
    	# Compute the velocity of agents
    	agent_v = []
    	agent_v.append([dp1_x, dp1_y])
    	agent_v.append([dp2_x, dp2_y])
    	agent_v = np.array(agent_v)
    	agent_v_angle = np.array([0.0, 0.0, 0.0])
    	
    	for i in range(0, 2):
    		if agent_v[i][0] > 0 and agent_v[i][1] >= 0:
    			agent_v_angle[i] = math.atan(math.tan(agent_v[i][1]/agent_v[i][0]))
    		elif agent_v[i][0] > 0 and agent_v[i][1] < 0:
    			agent_v_angle[i] = math.atan(math.tan(agent_v[i][1]/agent_v[i][0])) + 2*np.pi
    		elif agent_v[i][0] < 0 and agent_v[i][1] >= 0:
    			agent_v_angle[i] = math.atan(math.tan(agent_v[i][1]/agent_v[i][0])) + np.pi
    		elif agent_v[i][0] < 0 and agent_v[i][1] < 0:
    			agent_v_angle[i] = math.atan(math.tan(agent_v[i][1]/agent_v[i][0])) + np.pi
    		elif agent_v[i][0] == 0:
    			if agent_v[i][1] > 0:
    				agent_v_angle[i] = 1/2*np.pi
    			else:
    				agent_v_angle[i] = 3/2*np.pi
    	
    	# Compute local pose
    	# p1_p2
    	d = p2-p1
    	p1_p2_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p1_p2_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p1_p2_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p1_p2_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p1_p2_angle = 1/2*np.pi
    		else:
    			p1_p2_angle = 3/2*np.pi
    	# p1_e
    	d = e-p1
    	p1_e_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p1_e_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p1_e_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p1_e_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p1_e_angle = 1/2*np.pi
    		else:
    			p1_e_angle = 3/2*np.pi
    	# p2_p1
    	d = p1-p2
    	p2_p1_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p2_p1_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p2_p1_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p2_p1_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p2_p1_angle = 1/2*np.pi
    		else:
    			p2_p1_angle = 3/2*np.pi
    	# p2_e
    	d = e-p2
    	p2_e_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p2_e_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p2_e_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p2_e_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p2_e_angle = 1/2*np.pi
    		else:
    			p2_e_angle = 3/2*np.pi
    	
    	# Compute the distance to obstacle
    	# Pursuer No.1
    	d = ob1-p1
    	p1_ob1_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p1_ob1_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p1_ob1_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p1_ob1_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p1_ob1_angle = 1/2*np.pi
    		else:
    			p1_ob1_angle = 3/2*np.pi
    	d = ob2-p1
    	p1_ob2_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p1_ob2_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p1_ob2_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p1_ob2_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p1_ob2_angle = 1/2*np.pi
    		else:
    			p1_ob2_angle = 3/2*np.pi
    	# Pursuer No.2
    	d = ob1-p2
    	p2_ob1_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p2_ob1_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p2_ob1_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p2_ob1_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p2_ob1_angle = 1/2*np.pi
    		else:
    			p2_ob1_angle = 3/2*np.pi
    	d = ob2-p2
    	p2_ob2_r = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		p2_ob2_angle = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		p2_ob2_angle = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		p2_ob2_angle = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			p2_ob2_angle = 1/2*np.pi
    		else:
    			p2_ob2_angle = 3/2*np.pi
    	
    	self.steps_beyond_done = None
    	
    	# Get New Distance
    	New_distance = []
    	New_distance.append(p1_e_r)
    	New_distance.append(p2_e_r)
    	# print(New_distance)
    	
    	# Compute the distance to wall
    	# Pursuer No.1
    	p1_l1 = p1[0]
    	p1_l2 = 100 - p1[0]
    	p1_l3 = p1[1]
    	p1_l4 = 100 - p1[1]
    	# Pursuer No.2
    	p2_l1 = p2[0]
    	p2_l2 = 100 - p2[0]
    	p2_l3 = p2[1]
    	p2_l4 = 100 - p2[1]
    	
    	next_state = np.array([[p1[0], p1[1], agent_v_angle[0], p1_p2_r, p1_p2_angle, p1_e_r, p1_e_angle, p1_l1, p1_l2, p1_l3, p1_l4, p1_ob1_r, p1_ob1_angle, ob1_r, p1_ob2_r, p1_ob2_angle, ob2_r], [p2[0], p2[1], agent_v_angle[1], p2_p1_r, p2_p1_angle, p2_e_r, p2_e_angle, p2_l1, p2_l2, p2_l3, p2_l4, p2_ob1_r, p2_ob1_angle, ob1_r, p2_ob2_r, p2_ob2_angle, ob2_r]])
    	self.state = next_state
    	
    	success = (np.sqrt(np.sum(np.square(p1-e))) <= 2 or np.sqrt(np.sum(np.square(p2-e))) <= 2)
    	done = success or self.steps >= 599
    	done = bool(done)
    	
    	distance_difference = [New_distance[0]-last_distance[0], New_distance[1]-last_distance[1]]
    	# print(distance_difference)
    	
    	if done == 0:
    		exploration_pursuer_reward_1 = (-1)*distance_difference[0]*0.5
    		exploration_pursuer_reward_2 = (-1)*distance_difference[1]*0.5
    		self.steps += 1
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
    	return next_state, reward, done, info
    	
    def reset(self):
    	# Initial the high and low
    	pursuer_high= np.array([[100.0, 100.0, 2*np.pi, 100.0*(2**0.5), 2*np.pi, 100.0*(2**0.5), 2*np.pi, 100.0, 100.0, 100.0, 100.0, (80**2+80**2)**0.5, 2*np.pi, 15, (75**2+75**2)**0.5, 2*np.pi, 20], [100.0, 100.0, 2*np.pi, 100.0*(2**0.5), 2*np.pi, 100.0*(2**0.5), 2*np.pi, 100.0, 100.0, 100.0, 100.0, (80**2+80**2)**0.5, 2*np.pi, 15, (75**2+75**2)**0.5, 2*np.pi, 20]])
    	pursuer_low = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20]])
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
    	
    	
    	# Initial the p_p_r and p_p_angle
    	# p1_p2
    	d = np.array([pursuer_state[1][0], pursuer_state[1][1]])-np.array([pursuer_state[0][0], pursuer_state[0][1]])
    	pursuer_state[0][3] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[0][4] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[0][4] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[0][4] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[0][4] = 1/2*np.pi
    		else:
    			pursuer_state[0][4] = 3/2*np.pi
    	# p1_e
    	d = np.array([self.e_x, self.e_y])-np.array([pursuer_state[0][0], pursuer_state[0][1]])
    	pursuer_state[0][5] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[0][6] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[0][6] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[0][6] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[0][6] = 1/2*np.pi
    		else:
    			pursuer_state[0][6] = 3/2*np.pi
    	# p2_p1
    	d = np.array([pursuer_state[0][0], pursuer_state[0][1]])-np.array([pursuer_state[1][0], pursuer_state[1][1]])
    	pursuer_state[1][3] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[1][4] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[1][4] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[1][4] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[1][4] = 1/2*np.pi
    		else:
    			pursuer_state[1][4] = 3/2*np.pi
    	# p2_e
    	d = np.array([self.e_x, self.e_y])-np.array([pursuer_state[1][0], pursuer_state[1][1]])
    	pursuer_state[1][5] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[1][6] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[1][6] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[1][6] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[1][6] = 1/2*np.pi
    		else:
    			pursuer_state[1][6] = 3/2*np.pi
    	
    	# Initial the distance to wall
    	# Pursuer No.1
    	pursuer_state[0][7] = pursuer_state[0][0]
    	pursuer_state[0][8] = 100 - pursuer_state[0][0]
    	pursuer_state[0][9] = pursuer_state[0][1]
    	pursuer_state[0][10] = 100 - pursuer_state[0][1]
    	# Pursuer No.2
    	pursuer_state[1][7] = pursuer_state[1][0]
    	pursuer_state[1][8] = 100 - pursuer_state[1][0]
    	pursuer_state[1][9] = pursuer_state[1][1]
    	pursuer_state[1][10] = 100 - pursuer_state[1][1]
    	
    	# Initial the distance to obstacle
    	# Pursuer No.1
    	d = np.array([80, 80])-np.array([pursuer_state[0][0], pursuer_state[0][1]])
    	pursuer_state[0][11] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[0][12] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[0][12] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[0][12] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[0][12] = 1/2*np.pi
    		else:
    			pursuer_state[0][12] = 3/2*np.pi
    	d = np.array([25, 25])-np.array([pursuer_state[0][0], pursuer_state[0][1]])
    	pursuer_state[0][14] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[0][15] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[0][15] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[0][15] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[0][15] = 1/2*np.pi
    		else:
    			pursuer_state[0][15] = 3/2*np.pi
    	# Pursuer No.2
    	d = np.array([80, 80])-np.array([pursuer_state[1][0], pursuer_state[1][1]])
    	pursuer_state[1][11] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[1][12] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[1][12] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[1][12] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[1][12] = 1/2*np.pi
    		else:
    			pursuer_state[1][12] = 3/2*np.pi
    	d = np.array([25, 25])-np.array([pursuer_state[1][0], pursuer_state[1][1]])
    	pursuer_state[1][14] = np.sqrt(np.sum(np.square(d)))
    	if d[0] > 0 and d[1] >= 0:
    		pursuer_state[1][15] = math.atan(math.tan(d[1]/d[0]))
    	elif d[0] > 0 and d[1] < 0:
    		pursuer_state[1][15] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
    	elif d[0] < 0:
    		pursuer_state[1][15] = math.atan(math.tan(d[1]/d[0])) + np.pi
    	elif d[0] == 0:
    		if d[1] > 0:
    			pursuer_state[1][15] = 1/2*np.pi
    		else:
    			pursuer_state[1][15] = 3/2*np.pi
    	
    	states = [[0] * (17) for _ in range(2)]
    	for i in range(2):
    		for j in range(17):
    			states[i][j] = pursuer_state[i][j]
    	
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
    		
    		self.pursuer1_capture = rendering.make_circle(2, 30, filled=False)
    		self.p1ctrans = rendering.Transform()
    		self.pursuer1_capture.add_attr(self.p1ctrans)
    		self.pursuer1_capture.set_color(0, 1, 0)
    		
    		self.pursuer2 = rendering.make_circle(1)
    		self.p2trans = rendering.Transform()
    		self.pursuer2.add_attr(self.p2trans)
    		self.pursuer2.set_color(0, 1, 0)
    		
    		self.pursuer2_capture = rendering.make_circle(2, 30, filled=False)
    		self.p2ctrans = rendering.Transform()
    		self.pursuer2_capture.add_attr(self.p2ctrans)
    		self.pursuer2_capture.set_color(0, 1, 0)
    		
    		self.evader = rendering.make_circle(1)
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
            
def LineIntersectCircle(circle, p,p_new):
# p is the circle parameter, lsp and lep is the two end of the line
	x0,y0,r = circle
	x1,y1 = p
	x2,y2 = p_new
	x1 = x1 - x0
	y1 = y1 - y0
	x2 = x2 - x0
	y2 = y2 - y0
	a = y2 - y1
	b = x1 - x2
	c = x2*y1 - x1*y2
	x = -a*c/(a*a+b*b)
	y = -b*c/(a*a+b*b)
	d = r*r - c*c/(a*a+b*b)
	mult = (d / (a*a+b*b))**0.5
	ax = x + b * mult
	bx = x - b * mult
	ay = y - a * mult
	by = y + a * mult
	if abs(b) <= 1e-06:
		if ay <= max(y1, y2) and ay >= min(y1, y2):
			inp = [ax+x0, ay+y0]
		else:
			inp = [bx+x0, by+y0]
	elif abs(a) == 1e-06:
		if ax <= max(x1, x2) and ax >= min(x1, x2):
			inp = [ax+x0, ay+y0]
		else:
			inp = [bx+x0, by+y0]
	else:
		if ay <= max(y1, y2) and ay >= min(y1, y2):
			inp = [ax+x0, ay+y0]
		else:
			inp = [bx+x0, by+y0]
	return inp[0], inp[1]

def get_pose(P1, P2, P3, P):
	P4 = [0, 0]
	if P1[1] == P2[1]:
		P4[0] = P2[0]
		P4[1] = P3[1]
	elif abs(P1[0] - P2[0]) <= 1e-06:
		P4[0] = P3[0]
		P4[1] = P2[1]
	else:
		a1 = P2[1] - P1[1]
		b1 = P2[0] - P1[0]
		k1 = a1/b1
		c1 = P3[1] - k1*P3[0]
		k2 = -b1/a1
		c2 = P2[1] - k2*P2[0]
		P4[0] = (c2 - c1) / (k1 - k2)
		P4[1] = k1 * P4[0] + c1
	return P4[0]-P[0], P4[1]-P[1]
