import os
import math
import numpy as np
# import cv2

import sys
sys.path.append('/scratch/aakriti/complex_task_10rob_layout_newtask/warehouse/envs')
from Astar import astar_my


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
#693885 - tight_2
# 578275 - compact
# 677166 - tight
# 498346 - compact


#405555 - tight 2
#633980 - tight
#969029 - usc
# compact - 932150
# compact_2 - 299772

class WarehouseEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.agent_no = 10  #100
    self.tasks_no = 10  #6
    self.length = 3*(self.agent_no+1)+(6*self.tasks_no)
    
    self.action_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.uint8)
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
    self.f = open("seed_4data_usc_64.txt", 'w')
    self.RL_pos = open("plot_RL.txt", 'w')

    self.layout = np.loadtxt('./layouts/usc_64.cfg', skiprows=10)
    self.astar = Astar(self.layout)
    self.astar_my = astar_my('./layouts/usc_64.cfg')  #Astar(self.layout)


    self.closest_dist = 0
    self.cum_reward = 0      
    self._envStepCounter = 0

    ### sipp
    self.current_robot_state = np.zeros((self.agent_no, 3))
    count = 0
    while(count < self.agent_no):
      self.current_robot_state[count][0] =  abs(np.round(gauss(30, 10)))
      self.current_robot_state[count][1] =  abs(np.round(gauss(30, 10)))
      self.current_robot_state[count][2] =  2
      
      if(self.layout[int(self.current_robot_state[count][0])][int(self.current_robot_state[count][1])] == 0):
        count = count+1
    ### sipp     
    

    self.robot_positions = np.zeros((self.agent_no, 3))    
    count = 0
    while(count < self.agent_no):
      self.robot_positions[count][0] =  abs(np.round(gauss(10, 10)))
      self.robot_positions[count][1] =  abs(np.round(gauss(10, 10)))
      self.robot_positions[count][2] =  np.size(self.astar.run([int(self.current_robot_state[count,0]), int(self.current_robot_state[count,1])], [int(self.robot_positions[count, 0]), int(self.robot_positions[count, 1])]))/2
      
        
      if(self.layout[int(self.robot_positions[count][0])][int(self.robot_positions[count][1])] == 0):
        if count == 0:
          self.RL_pos.write(str((self.robot_positions[0][0], self.robot_positions[0][1]))+",")
        else: 
          posi = list(self.astar_my.astar((int(self.current_robot_state[count,0]), int(self.current_robot_state[count,1])), (int(self.robot_positions[count, 0]), int(self.robot_positions[count, 1]))))
          # print(posi)
          for p in posi:
            self.RL_pos.write(str(p)+",")
          
        self.RL_pos.write("\n")
        count = count+1

    
    self.RL_pos.close()

    self.robot_positions[0][2] = 0
    self.current_robot_state[0, :] =  self.robot_positions[0,:]
    self.current_robot_state[0,2] = 0

    np.savetxt(self.f, self.current_robot_state)
    np.savetxt(self.f, self.robot_positions)

    self.current_robot_state[0, :] =  self.robot_positions[0,:]
    self.current_robot_state[0,2] = 0

    self.goal_states = np.zeros((self.agent_no, 4))
    self.goal_states[:,2:4] = np.copy(self.robot_positions[:,0:2])

    self.total_time = 0
    #sipp end

    self.min_r = 0
    self.old_min_r = 1  

    self.tasks_input = np.zeros((self.tasks_no, 6))
    count = 0 
    while(count != self.tasks_no):   
      for i in range(2):
        num = random.randint(0,4)
        if (num == 0):  
          self.tasks_input[count, 2*i] = abs(np.round(gauss(11,5)))
          self.tasks_input[count, 2*i+1] = abs(np.round(gauss(10,5)))
        elif (num == 1): 
          self.tasks_input[count, 2*i] = abs(np.round(gauss(29,5)))
          self.tasks_input[count, 2*i+1] = abs(np.round(gauss(29,5)))
        elif (num == 2): 
          self.tasks_input[count, 2*i] = abs(np.round(gauss(48,5)))
          self.tasks_input[count, 2*i+1] = abs(np.round(gauss(16,5)))
        elif (num == 3): 
          self.tasks_input[count, 2*i] = abs(np.round(gauss(48,5)))
          self.tasks_input[count, 2*i+1] = abs(np.round(gauss(29,5)))
        elif (num == 4): 
          self.tasks_input[count, 2*i] = abs(np.round(gauss(4,5)))
          self.tasks_input[count, 2*i+1] = abs(np.round(gauss(54,5)))      

      if((self.tasks_input[count, 0] > 59) or (self.tasks_input[count, 1] > 59) or (self.tasks_input[count, 2] > 59) or (self.tasks_input[count, 3] > 59)): continue
      

      self.tasks_input[count, 4] = np.size(self.astar.run([int(self.robot_positions[self.min_r][0]), int(self.robot_positions[self.min_r][1])], [int(self.tasks_input[count, 0]), int(self.tasks_input[count, 1])]))/2  #+ gauss(0, 0.25)
      self.tasks_input[count, 5] = np.size(self.astar.run([int(self.tasks_input[count, 0]), int(self.tasks_input[count, 1])], [int(self.tasks_input[count, 2]), int(self.tasks_input[count, 3])]))/2
      
      
      if((self.layout[int(self.tasks_input[count,0])][int(self.tasks_input[count,1])] == 0) and (self.layout[int(self.tasks_input[count, 2])][int(self.tasks_input[count, 3])] == 0)): 
        count = count + 1

    np.savetxt(self.f, self.tasks_input)
    self._observation = np.reshape(np.concatenate((self.tasks_input.flatten().reshape((6*self.tasks_no,1)), self.robot_positions.flatten().reshape((3*self.agent_no,1)), self.robot_positions[self.min_r].flatten().reshape((3,1))), axis=0), self.length) #69 339# np.array([np.reshape(self.sending_tasks, (64,64)), np.reshape(self.sending_pos, (64,64))]) #, np.reshape(self.sending_selected_t, (64,64))])
    
    self.fGreedy = open("actions_RL.txt", "w")
    
    return self._observation 

 
  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _compute_observation(self, action):
    # random.seed(4)
    # np.random.seed(4)

    start_end = np.round(self.tasks_input[action])
    start = np.round(self.robot_positions[self.min_r][0:2])
    task_time = np.size(self.astar.run([int(start[0]), int(start[1])], [int(start_end[0]), int(start_end[1])]))/2 + self.tasks_input[action][5] # ((start[0]-start_end[0])**2 + (start[1]-start_end[1])**2)**0.5 + ((start_end[0]-start_end[2])**2 + (start_end[1]-start_end[3])**2)**0.5
    
    # for video
    np.savetxt(self.fGreedy, np.array([[self.total_time, action]]))

    f = open("plot_RL.txt", 'r')
    data = f.readlines()
    f.close()
    data[self.min_r] = data[self.min_r].rstrip("\n")

    posi = list(self.astar_my.astar((int(start[0]), int(start[1])), (int(start_end[0]), int(start_end[1])))) #self.astar.run([int(start[0]), int(start[1])], [int(start_end[0]), int(start_end[1])])[1:]
    for p in posi:      
      data[self.min_r] = data[self.min_r] + str(p) + ","
    
    posi = list(self.astar_my.astar((int(start_end[0]), int(start_end[1])), (int(start_end[2]), int(start_end[3])))) #self.astar.run([int(start_end[0]), int(start_end[1])], [int(start_end[2]), int(start_end[3])])[1:]
    for p in posi:      
      data[self.min_r] = data[self.min_r] + str(p) + ","


    data[self.min_r] = data[self.min_r] + "\n"
    f = open("plot_RL.txt", 'w')
    f.writelines(data)
    f.close()

    # end for video
            
    
    
    # test_index = np.zeros(self.tasks_no)
    # for i in range(self.tasks_no):
    #   start_end_test = np.round(self.tasks_input[i]).astype(int)
    #   test_index[i] = np.size(self.astar.run([int(start[0]), int(start[1])], [int(start_end_test[0]), int(start_end_test[1])]))/2 #((start[0]-start_end_test[0])**2 + (start[1]-start_end_test[1])**2)**0.5
    

    # self.closest_dist = self.closest_dist + test_index[np.argmin(test_index)]
    # # print("closest dist reward ", self.closest_dist)
    
    # start_end_test = np.round(self.tasks_input[np.argmin(test_index)]).astype(int)
    # test_time_testing = test_index[np.argmin(test_index)] + self.tasks_input[np.argmin(test_index)][5]
    
    self.robot_positions[self.min_r][0:2] = self.tasks_input[action][2:4]
    self.robot_positions[self.min_r][2] = task_time + gauss(0.25, 0.25)

    self.old_min_r = self.min_r
    self.min_r = np.argmin(self.robot_positions[:,2])
    self.total_time = self.total_time + self.robot_positions[self.min_r,2]

    self.robot_positions[:,2] = self.robot_positions[:,2] -  self.robot_positions[self.min_r][2]
    
    
    count = 0
    while(count != 1):
      for i in range(2):
        num = random.randint(0,4)

        if (num == 0): 
          self.tasks_input[action, 2*i] = abs(np.round(gauss(11,5)))
          self.tasks_input[action, 2*i+1] = abs(np.round(gauss(10,5)))
        elif (num == 1): 
          self.tasks_input[action, 2*i] = abs(np.round(gauss(29,5)))
          self.tasks_input[action, 2*i+1] = abs(np.round(gauss(29,5)))
        elif (num == 2): 
          self.tasks_input[action, 2*i] = abs(np.round(gauss(48,5)))
          self.tasks_input[action, 2*i+1] = abs(np.round(gauss(16,5)))
        elif (num == 3): 
          self.tasks_input[action, 2*i] = abs(np.round(gauss(48,5)))
          self.tasks_input[action, 2*i+1] = abs(np.round(gauss(29,5)))
        elif (num == 4): 
          self.tasks_input[action, 2*i] = abs(np.round(gauss(4,5)))
          self.tasks_input[action, 2*i+1] = abs(np.round(gauss(54,5)))       
      
      if((self.tasks_input[action, 0] > 59) or (self.tasks_input[action, 1] > 59) or (self.tasks_input[action, 2] > 59) or (self.tasks_input[action, 3] > 59)): continue
      
      if((self.layout[int(self.tasks_input[action, 0])][int(self.tasks_input[action, 1])] == 0) and (self.layout[int(self.tasks_input[action, 2])][int(self.tasks_input[action, 3])] == 0)): 
        count = count + 1
    
    self.tasks_input[action, 5] = np.size(self.astar.run([int(self.tasks_input[action, 0]), int(self.tasks_input[action, 1])], [int(self.tasks_input[action, 2]), int(self.tasks_input[action, 3])]))/2   #((self.tasks_input[action][0] - self.tasks_input[action][2])**2 + (self.tasks_input[action][1] - self.tasks_input[action][3])**2)**0.5
    
    np.savetxt(self.f, self.tasks_input[action].reshape((1,6)))
     

    count = 0 
    while(count != self.tasks_no):      
      self.tasks_input[count][4] = np.size(self.astar.run([int(self.robot_positions[self.min_r][0]), int(self.robot_positions[self.min_r][1])], [int(self.tasks_input[count][0]), int(self.tasks_input[count][1])]))/2  #((self.robot_positions[self.min_r][0] - self.tasks_input[count][0])**2 + (self.robot_positions[self.min_r][1] - self.tasks_input[count][1])**2)**0.5 #+ gauss(0, 0.25)
      count = count+1
      
    np.random.shuffle(self.tasks_input) 

    self._observation = np.reshape(np.concatenate((self.tasks_input.flatten().reshape((6*self.tasks_no,1)), self.robot_positions.flatten().reshape((3*self.agent_no,1)), self.robot_positions[self.min_r].flatten().reshape((3,1))), axis=0), self.length) #69 339#np.array([np.reshape(self.sending_tasks, (64,64)), np.reshape(self.sending_pos, (64,64))]) #, np.reshape(self.sending_selected_t, (64,64))])
    return self._observation

  def _compute_reward(self, action):  
    start_end = np.round(self.tasks_input[action]) 
    start = np.round(self.robot_positions[self.min_r][0:2])
    half_reward =  -np.size(self.astar.run([int(start[0]), int(start[1])], [int(start_end[0]), int(start_end[1])]))/2 # -((start[0]-start_end[0])**2 + (start[1]-start_end[1])**2)**0.5
    # paper_reward = ((start_end[0]-start_end[2])**2 + (start_end[1]-start_end[3])**2)**0.5

    self.cum_reward = self.cum_reward - half_reward
    print("cumulative reward ", self.cum_reward)
    return half_reward
     
 
  def _compute_done(self):
    # if(np.max(self.robot_time) > 150):
    #   return True
    if((self._envStepCounter ) > 500):  #600
      return True
    return False   

  def render(self, action, mode='human', close=False):
    return 0
