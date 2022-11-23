import numpy as np

agent_no = 100
tasks_no = 10

# Greedy 

robots = np.loadtxt("seed_4data.txt", max_rows=agent_no).astype(float)
tasks_load = np.loadtxt("seed_4data.txt", skiprows=agent_no).astype(float)
tasks = tasks_load[:,0:4]

def robot_dist(robot_minr, tasks_10):
    dist = np.zeros(tasks_no)
    for i in range(tasks_no):
        dist[i] = ((robot_minr[0] - tasks_10[i, 0])**2 + (robot_minr[1] - tasks_10[i, 1])**2)**0.5 
    
    return dist

min_r = 0

tasks_10 = tasks[0:tasks_no]
greedy_ttd = 0

for i in range(500):
    
    rob_dist = robot_dist(robots[min_r], tasks_10)
    chosen_task = np.argmin(rob_dist)
    greedy_ttd = greedy_ttd + rob_dist[chosen_task]
    # print(tasks_10, rob_dist, chosen_task, min_r)

    robots[min_r, 2] = ((robots[min_r, 0] - tasks_10[chosen_task, 0])**2 + (robots[min_r, 1] - tasks_10[chosen_task, 1])**2)**0.5 + ((tasks_10[chosen_task, 0] - tasks_10[chosen_task, 2])**2 + (tasks_10[chosen_task, 1] - tasks_10[chosen_task, 3])**2)**0.5
    robots[min_r, 0] = tasks_10[chosen_task, 2]
    robots[min_r, 1] = tasks_10[chosen_task, 3]
    
    # print(robots[:, 2])
    min_r = np.argmin(robots[:, 2])
    robots[:, 2] = robots[:, 2] - robots[min_r, 2]

    tasks_10[chosen_task, 0] = tasks_load[i+tasks_no, 0]
    tasks_10[chosen_task, 1] = tasks_load[i+tasks_no, 1]
    tasks_10[chosen_task, 2] = tasks_load[i+tasks_no, 2]
    tasks_10[chosen_task, 3] = tasks_load[i+tasks_no, 3]
print("Greedy TTD", greedy_ttd, rob_dist[chosen_task])
    


# Regret based task selection

def find_closest_robot_dist(robots, tasks):
    dist = np.zeros(agent_no)
    for i in range(agent_no):
        dist[i] = ((robots[i,0] - tasks[0])**2 + (robots[i,1] - tasks[1])**2)**0.5 
    return np.min(dist)

    

robots = np.loadtxt("seed_4data.txt", max_rows=agent_no).astype(float)
tasks_load = np.loadtxt("seed_4data.txt", skiprows=agent_no).astype(float)
tasks = tasks_load[:,0:4]

min_r = 0

tasks_10 = tasks[0:tasks_no]
regret_ttd = 0
closest_robot_dist = np.zeros(tasks_no)

for i in range(500):    
    for j in range(tasks_no):
        closest_robot_dist[j] = find_closest_robot_dist(robots, tasks_10[j])
        rob_dist[j] = ((robots[min_r, 0] - tasks_10[j, 0])**2 + (robots[min_r, 1] - tasks_10[j, 1])**2)**0.5

    chosen_task = np.argmax(closest_robot_dist - rob_dist)

    regret_ttd = regret_ttd + rob_dist[chosen_task]

    robots[min_r, 2] = ((robots[min_r, 0] - tasks_10[chosen_task, 0])**2 + (robots[min_r, 1] - tasks_10[chosen_task, 1])**2)**0.5 + ((tasks_10[chosen_task, 0] - tasks_10[chosen_task, 2])**2 + (tasks_10[chosen_task, 1] - tasks_10[chosen_task, 3])**2)**0.5

    robots[min_r, 0] = tasks_10[chosen_task, 2]
    robots[min_r, 1] = tasks_10[chosen_task, 3]
    
    min_r = np.argmin(robots[:, 2])
    robots[:, 2] = robots[:, 2] - robots[min_r, 2]

    tasks_10[chosen_task, 0] = tasks_load[i+tasks_no, 0]
    tasks_10[chosen_task, 1] = tasks_load[i+tasks_no, 1]
    tasks_10[chosen_task, 2] = tasks_load[i+tasks_no, 2]
    tasks_10[chosen_task, 3] = tasks_load[i+tasks_no, 3]
print("Regret TTD", regret_ttd, rob_dist[chosen_task])


