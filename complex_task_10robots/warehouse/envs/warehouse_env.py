import os
import math
import numpy as np
# import cv2
from astar_python.astar import Astar

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image
import random
# import time
# random.seed(time.clock())
from random import gauss

#1. change to window method by keeping the input as 200 tasks only 

class WarehouseEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.agent_no = 10  #10
    self.tasks_no = 10  #20
    self.action_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.uint8)
    self.length = 3*(self.agent_no+1)+(6*self.tasks_no)
    self.observation_space = spaces.Box(low=0, high=15, shape=(self.length, 1), dtype=np.float)   #339

    self._seed()

  def step(self, action):
    reward = self._compute_reward(action)
    self._observation = self._compute_observation(action) 

    done = self._compute_done()
    self._envStepCounter += 1
    return self._observation, reward, done, {}

  def reset(self):
    # random.seed(4)
    # np.random.seed(4)
    
    self.f = open("seed_4data.txt", 'w')

    self.closest_dist = 0
    self.cum_reward = 0
      
    self._envStepCounter = 0
    self.simulation_time = 0

    self.robot_positions = np.zeros((self.agent_no, 3))

    for count in range(0,self.agent_no):
      self.robot_positions[count][0] = np.round(gauss(10, 10))
      self.robot_positions[count][1] = np.round(gauss(10, 10))
      self.robot_positions[count][2] = np.round(gauss(20, 10))

    self.robot_positions[0][2] = 0

    np.savetxt(self.f, self.robot_positions)
    
    self.robot_test_time = np.copy(self.robot_positions[:,2])
    self.robot_time = np.copy(self.robot_positions[:,2])

    self.min_r = 0
    self.old_min_r = 1  

    self.tasks_input = np.zeros((self.tasks_no, 6))
    count = 0 
    while(count != self.tasks_no):
      for i in range(2):
        num = random.randint(0,4)
        if (num == 0): 
          self.tasks_input[count, 2*i] = np.round(gauss(10,5))
          self.tasks_input[count, 2*i+1] = np.round(gauss(10,5))
        elif (num == 1): 
          self.tasks_input[count, 2*i] = np.round(gauss(40,5))
          self.tasks_input[count, 2*i+1] = np.round(gauss(40,5))
        elif (num == 2): 
          self.tasks_input[count, 2*i] = np.round(gauss(70,5))
          self.tasks_input[count, 2*i+1] = np.round(gauss(40,5))
        elif (num == 3): 
          self.tasks_input[count, 2*i] = np.round(gauss(25,5))
          self.tasks_input[count, 2*i+1] = np.round(gauss(90,5))
        elif (num == 4): 
          self.tasks_input[count, 2*i] = np.round(gauss(70,5))
          self.tasks_input[count, 2*i+1] = np.round(gauss(20,5))      

      self.tasks_input[count, 4] = ((self.robot_positions[self.min_r][0] - self.tasks_input[count][0])**2 + (self.robot_positions[self.min_r][1] - self.tasks_input[count][1])**2)**0.5      # distance of free robot 
      self.tasks_input[count, 5] = ((self.tasks_input[count][0] - self.tasks_input[count][2])**2 + (self.tasks_input[count][1] - self.tasks_input[count][3])**2)**0.5   # task length
      
      count = count+1 
    np.savetxt(self.f, self.tasks_input)

    # print("greedy", np.max(self.robot_test_time))
    # print("ours: ", np.max(self.robot_time))

    self._observation = np.reshape(np.concatenate((self.tasks_input.flatten().reshape((6*self.tasks_no,1)), self.robot_positions.flatten().reshape((3*self.agent_no,1)), self.robot_positions[self.min_r].flatten().reshape((3,1))), axis=0), self.length)
    
    return self._observation 

 
  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _compute_observation(self, action):
    # random.seed(self._envStepCounter)
    # np.random.seed(4)
    
    
    start_end = np.round(self.tasks_input[action])
    start = np.round(self.robot_positions[self.min_r][0:2])
    task_time = ((start[0]-start_end[0])**2 + (start[1]-start_end[1])**2)**0.5 + ((start_end[0]-start_end[2])**2 + (start_end[1]-start_end[3])**2)**0.5
    
    test_index = np.zeros(self.tasks_no)
    for i in range(self.tasks_no):
      start_end_test = np.round(self.tasks_input[i]).astype(int)
      test_index[i] = ((start[0]-start_end_test[0])**2 + (start[1]-start_end_test[1])**2)**0.5 #+ ((start_end_test[0]-start_end_test[2])**2 + (start_end_test[1]-start_end_test[3])**2)**0.5
    
    # print("test index ", np.argmin(test_index), test_index[np.argmin(test_index)], "action ", action, test_index[action])

    self.closest_dist = self.closest_dist + test_index[np.argmin(test_index)]
    # print("closest dist reward ", self.closest_dist)
    
    start_end_test = np.round(self.tasks_input[np.argmin(test_index)]).astype(int)
    test_time_testing = ((start[0]-start_end_test[0])**2 + (start[1]-start_end_test[1])**2)**0.5 + ((start_end_test[0]-start_end_test[2])**2 + (start_end_test[1]-start_end_test[3])**2)**0.5
    
    self.robot_positions[self.min_r][0:2] = self.tasks_input[action][2:4]
    self.robot_positions[self.min_r][2] = task_time + gauss(0, 0.25)

    self.old_min_r = self.min_r
    self.min_r = np.argmin(self.robot_positions[:,2])
    self.robot_positions[:,2] = self.robot_positions[:,2] -  self.robot_positions[self.min_r][2]
    
    self.robot_test_time[self.old_min_r] = self.robot_test_time[self.old_min_r] + test_time_testing #- self.robot_test_time[self.min_r]
    self.robot_time[self.old_min_r] = self.robot_time[self.old_min_r] + task_time

    # print("Times!!!")
    # print("greedy", np.max(self.robot_test_time))
    # print("ours: ", np.max(self.robot_time))
    
    # print("task before", self.tasks_input)
    new_task = np.zeros((1,6))
    for i in range(2):
      num = random.randint(0,4)
      if (num == 0): 
        new_task[0][2*i] = abs(np.round(gauss(10,5)))
        new_task[0][2*i+1] = abs(np.round(gauss(10,5)))
      elif (num == 1): 
        new_task[0][2*i] = abs(np.round(gauss(40,5)))
        new_task[0][2*i+1] = abs(np.round(gauss(40,5)))
      elif (num == 2): 
        new_task[0][2*i] = abs(np.round(gauss(70,5)))
        new_task[0][2*i+1] = abs(np.round(gauss(40,5)))
      elif (num == 3): 
        new_task[0][2*i] = abs(np.round(gauss(25,5)))
        new_task[0][2*i+1] = abs(np.round(gauss(90,5)))
      elif (num == 4): 
        new_task[0][2*i] = abs(np.round(gauss(70,5)))
        new_task[0][2*i+1] = abs(np.round(gauss(20,5)))      

    
    new_task[0][5] = ((self.tasks_input[action][0] - self.tasks_input[action][2])**2 + (self.tasks_input[action][1] - self.tasks_input[action][3])**2)**0.5
    np.savetxt(self.f, new_task)
    self.tasks_input[action] = new_task
    
    
    count = 0 
    while(count != self.tasks_no):      
      self.tasks_input[count][4] = ((self.robot_positions[self.min_r][0] - self.tasks_input[count][0])**2 + (self.robot_positions[self.min_r][1] - self.tasks_input[count][1])**2)**0.5 
      count = count+1
      
    # print("task after", new_task[0], self.tasks_input)    

      
    np.random.shuffle(self.tasks_input) 

    self._observation = np.reshape(np.concatenate((self.tasks_input.flatten().reshape((6*self.tasks_no,1)), self.robot_positions.flatten().reshape((3*self.agent_no,1)), self.robot_positions[self.min_r].flatten().reshape((3,1))), axis=0), self.length)
    return self._observation

  def _compute_reward(self, action):  
    # if (self.tasks_input[action[0]][3]>0): return 2 #* (0.99**(self.simulation_time - self.prev_simulation_time))
    # else: return 3 #* (0.99**(self.simulation_time - self.prev_simulation_time))
    start_end = np.round(self.tasks_input[action]) 
    start = np.round(self.robot_positions[self.min_r][0:2])
    half_reward = -((start[0]-start_end[0])**2 + (start[1]-start_end[1])**2)**0.5
    paper_reward = ((start_end[0]-start_end[2])**2 + (start_end[1]-start_end[3])**2)**0.5
    
    self.cum_reward = self.cum_reward - half_reward
    print("cum_rewardxxx", self.cum_reward) 
    return half_reward
     
 
  def _compute_done(self):
    # if(np.max(self.robot_time) > 150):
    #   return True
    if((self._envStepCounter ) > 600):  #600
      return True
    return False   

  def render(self, action, mode='human', close=False):
    return 0

