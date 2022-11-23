import numpy as np

import random
from random import gauss

import sys
sys.path.append('/scratch/aakriti/complex_task_10rob_layout_newtask/warehouse/envs')
from Astar import astar_my
astar_my = astar_my('./layouts/usc_64.cfg') 


from astar_python.astar import Astar
layout = np.loadtxt('./layouts/usc_64.cfg', skiprows=10)
astar = Astar(layout)





agent_no = 10 
tasks_no = 10

current_robot_state = np.loadtxt("seed_4data_usc_64.txt", max_rows=agent_no).astype(float)
robots = np.loadtxt("seed_4data_usc_64.txt", skiprows=agent_no, max_rows=agent_no).astype(float)
tasks_load = np.loadtxt("seed_4data_usc_64.txt", skiprows=2*agent_no).astype(float)
tasks = tasks_load[:,0:4]

RL_pos = open("plot_greedy.txt", 'w')

count = 0
while(count < agent_no):
    if count == 0:
        RL_pos.write(str((robots[0][0], robots[0][1]))+",")
        # print(str((robots[0][0], robots[0][1])))
    else:
        posi = list(astar_my.astar((int(current_robot_state[count, 0]), int(current_robot_state[count, 1])), (int(robots[count, 0]), int(robots[count, 1])))) 
        for p in posi:
            RL_pos.write(str(p)+",")    
    RL_pos.write("\n")  
    count = count+1  

RL_pos.close()


fGreedy = open("actions_greedy.txt", "w")
total_time = 0

def robot_dist(robot_minr, tasks_10):
    dist = np.zeros(tasks_no)
    for i in range(tasks_no):
        dist[i] = np.size(astar.run([int(robot_minr[0]), int(robot_minr[1])], [int(tasks_10[i, 0]), int(tasks_10[i, 1])]))/2 
    
    return dist

min_r = 0
tasks_10 = tasks[0:tasks_no]
greedy_ttd = 0


for i in range(500):
    rob_dist = robot_dist(robots[min_r], tasks_10)
    chosen_task = np.argmin(rob_dist)
    greedy_ttd = greedy_ttd + rob_dist[chosen_task]

    #for video
    np.savetxt(fGreedy, np.array([[total_time, chosen_task]]))

    start_end = np.round(tasks_10[chosen_task])
    start = np.round(robots[min_r][0:2])
    
    f = open("plot_greedy.txt", 'r')
    data = f.readlines()
    f.close()
    data[min_r] = data[min_r].rstrip("\n")

    posi = list(astar_my.astar((int(start[0]), int(start[1])), (int(start_end[0]), int(start_end[1])))) #astar.run([int(start[0]), int(start[1])], [int(start_end[0]), int(start_end[1])])[1:]
    for p in posi:      
      data[min_r] = data[min_r] + str(p) + ","
    
    posi = list(astar_my.astar((int(start_end[0]), int(start_end[1])), (int(start_end[2]), int(start_end[3])))) #astar.run([int(start_end[0]), int(start_end[1])], [int(start_end[2]), int(start_end[3])])[1:]
    for p in posi:      
      data[min_r] = data[min_r] + str(p) + ","

    data[min_r] = data[min_r] + "\n"
    f = open("plot_greedy.txt", 'w')
    f.writelines(data)
    f.close()

    # end for video
    

    robots[min_r, 2] = np.size(astar.run([int(robots[min_r, 0]), int(robots[min_r, 1])], [int(tasks_10[chosen_task, 0]), int(tasks_10[chosen_task, 1])]))/2  +  np.size(astar.run([int(tasks_10[chosen_task, 0]), int(tasks_10[chosen_task, 1])], [int(tasks_10[chosen_task, 2]), int(tasks_10[chosen_task, 3])]))/2

    robots[min_r, 0] = tasks_10[chosen_task, 2]
    robots[min_r, 1] = tasks_10[chosen_task, 3]
    
    min_r = np.argmin(robots[:, 2])
    total_time = total_time + robots[min_r, 2]
    robots[:, 2] = robots[:, 2] - robots[min_r, 2]

    tasks_10[chosen_task, 0] = tasks_load[i+tasks_no, 0]
    tasks_10[chosen_task, 1] = tasks_load[i+tasks_no, 1]
    tasks_10[chosen_task, 2] = tasks_load[i+tasks_no, 2]
    tasks_10[chosen_task, 3] = tasks_load[i+tasks_no, 3]

print("Greedy TTD", i, greedy_ttd)    

# Regret based task selection

def find_closest_robot_dist(robots, tasks):
    dist = np.zeros(agent_no)
    for i in range(agent_no):
        dist[i] = np.size(astar.run([int(robots[i,0]), int(robots[i,1])], [int(tasks[0]), int(tasks[0])]))/2 
    return np.min(dist)

    

rrobots = np.loadtxt("seed_4data_usc_64.txt", skiprows=agent_no, max_rows=agent_no).astype(float)
tasks_load = np.loadtxt("seed_4data_usc_64.txt", skiprows=2*agent_no).astype(float)
tasks = tasks_load[:,0:4]

min_r = 0

tasks_10 = tasks[0:tasks_no]
regret_ttd = 0
closest_robot_dist = np.zeros(tasks_no)

for i in range(500):    
    for j in range(tasks_no):
        closest_robot_dist[j] = find_closest_robot_dist(robots, tasks_10[j])
        rob_dist[j] = np.size(astar.run([int(robots[min_r, 0]), int(robots[min_r, 1])], [int(tasks_10[j, 0]), int(tasks_10[j, 1])]))/2 

    chosen_task = np.argmax(closest_robot_dist - rob_dist)

    regret_ttd = regret_ttd + rob_dist[chosen_task]

    robots[min_r, 2] = np.size(astar.run([int(robots[min_r, 0]), int(robots[min_r, 1])], [int(tasks_10[chosen_task, 0]), int(tasks_10[chosen_task, 1])]))/2  +  np.size(astar.run([int(tasks_10[chosen_task, 0]), int(tasks_10[chosen_task, 1])], [int(tasks_10[chosen_task, 2]), int(tasks_10[chosen_task, 3])]))/2 

    robots[min_r, 0] = tasks_10[chosen_task, 2]
    robots[min_r, 1] = tasks_10[chosen_task, 3]
    
    min_r = np.argmin(robots[:, 2])
    robots[:, 2] = robots[:, 2] - robots[min_r, 2]

    tasks_10[chosen_task, 0] = tasks_load[i+tasks_no, 0]
    tasks_10[chosen_task, 1] = tasks_load[i+tasks_no, 1]
    tasks_10[chosen_task, 2] = tasks_load[i+tasks_no, 2]
    tasks_10[chosen_task, 3] = tasks_load[i+tasks_no, 3]
print("Regret TTD", regret_ttd, rob_dist[chosen_task])


