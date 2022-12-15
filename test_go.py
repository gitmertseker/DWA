import math
import numpy as np
from DWA import Path,Obstacle,RobotState,Robot,Costmap
import matplotlib.pyplot as plt
import seaborn as sns
import pytest

sns.set()

init_x = 0.4
init_y = 0.4
init_theta = 0

init_v = 0
init_w =0

goal_x = 0
goal_y = 0.3

def go(init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y,
        filename= "local_costmap_copy.txt",
        min_v=0,max_v=0.1,min_w=-math.pi/4,max_w=math.pi/4,
        max_a_v=0.2,max_a_w=90*math.pi/180,delta_v=0.05,
        delta_w=np.deg2rad(18/4),dt=0.1,n=30,obs_sensitivity=0.5,goal_region = 0.02
        ):

    costmap = Costmap()
    cm = costmap.read_costmap(filename)
    cm = [[0 for i in row] for row in cm]   # empty costmap
    cm_rev = costmap.cm_rev(cm)
    cm_rev2 = costmap.cm_norm(cm_rev)
    obstacles = costmap.find_obstacles(cm_rev2)

    robot = Robot(cm_rev2,min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,obs_sensitivity)
    state = RobotState(init_x,init_y,init_theta,init_v,init_w)

    goal_Flag = False

    while not goal_Flag:
        
        paths,opt_path = robot.calc_opt_traj(goal_x,goal_y,state,obstacles,goal_region)

        # velocity commands
        opt_v = opt_path.v   
        opt_w = opt_path.w 
        #print("Optimal velocities are: ({},{})".format((opt_v),(opt_w)))
        state.update_state(opt_v,opt_w,dt)
 
        dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)
        if dis_to_goal < goal_region:
            goal_Flag = True

    print("Done")
    return state.traj(),robot.obs_pos_trial(obstacles),goal_Flag

@pytest.mark.parametrize("plt_name, init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y", [
    ("test_1",0.5, 0.5, 0, 0, 0, 0.1, 0.5),
    ("test_2",0.5, 0.5, 0, 0, 0, 0.9, 0.5),
    ("test_3",0.5, 0.5, 0, 0, 0, 0.5, 0.1),
    ("test_4",0.5, 0.5, 0, 0, 0, 0.5, 0.9),
    ("test_5",0.5, 0.5, 0, 0, 0, 0.1, 0.1),
    ("test_6",0.5, 0.5, 0, 0, 0, 0.1, 0.9),
    ("test_7",0.5, 0.5, 0, 0, 0, 0.9, 0.1),
    ("test_8",0.5, 0.5, 0, 0, 0, 0.9, 0.9)
])    
def test_dwa_wo_obs(plt, plt_name, init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y):
    [x_traj,y_traj], [obs_x, obs_y],goal_state = go(init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y)
    assert goal_state == True
    plt.plot(x_traj,y_traj, "-r")
    plt.plot(init_x, init_y, "xr")
    plt.plot(goal_x, goal_y, "xb")
    plt.plot(obs_x, obs_y, "ok")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.saveas = plt_name + ".png"
