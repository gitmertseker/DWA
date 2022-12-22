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

def get_costmap(filename,orig_px=30):
    costmap = Costmap()
    cm=np.zeros((orig_px*2,orig_px*2))
    if filename!="":
        cm[orig_px-20:orig_px+20,orig_px-20:orig_px+20]=costmap.read_costmap(filename)
    cm_rev = costmap.cm_rev(cm)
    cm_rev2 = costmap.cm_norm(cm_rev)
    obstacles = costmap.find_obstacles(cm_rev2)
    return cm_rev2,obstacles

def go(init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y,
        heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,
        costmap, obstacles,
        min_v=0,max_v=0.1,min_w=-math.pi/4,max_w=math.pi/4,
        max_a_v=0.2,max_a_w=90*math.pi/180,delta_v=0.05,
        delta_w=np.deg2rad(18/4),dt=0.1,n=30,obs_sensitivity=0.5,goal_region = 0.02,orig_px=30
        ):

    robot = Robot(costmap,min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,
                    heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,orig_px)
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

def plot_arrow(x, y, yaw, length=0.2, width=0.04):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            head_length=width, head_width=width)
    plt.plot(x, y)

@pytest.mark.wo_obstacle
@pytest.mark.parametrize("plt_name, init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y, heading_cost_weight, obstacle_cost_weight, velocity_cost_weight", [
    ("test_1",0, 0, 0, 0, 0, -0.5, 0.0, 3.1, 1, 1),
    ("test_2",0, 0, 0, 0, 0, 0.5, 0.0, 3.1, 1, 1),
    ("test_3",0, 0, 0, 0, 0, 0.0, -0.5, 3.1, 1, 1),
    ("test_4",0, 0, 0, 0, 0, 0.0, 0.5, 3.1, 1, 1),
    ("test_5",0, 0, 0, 0, 0, -0.5, -0.5, 3.1, 1, 1),
    ("test_6",0, 0, 0, 0, 0, -0.5, 0.5, 3.1, 1, 1),
    ("test_7",0, 0, 0, 0, 0, 0.5, -0.5, 3.1, 1, 1),
    ("test_8",0, 0, 0, 0, 0, 0.5, 0.5, 3.1, 1, 1)
])    
def test_dwa_wo_obs(plt, plt_name, init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y,
                    heading_cost_weight,obstacle_cost_weight,velocity_cost_weight):
    cm,obstacles = get_costmap("")
    [x_traj,y_traj], [obs_x, obs_y],goal_state = go(init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y,
                                                heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,
                                                cm, obstacles)
    assert goal_state == True
    if goal_state:
        plt.plot(x_traj,y_traj, "-r")
    plt.plot(init_x, init_y, "xr")
    plt.plot(goal_x, goal_y, "xb")
    plt.plot(obs_x, obs_y, "ok")
    plot_arrow(init_x, init_y, init_theta)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.saveas = plt_name + ".png"

@pytest.mark.with_obstacle
@pytest.mark.parametrize("plt_name, init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y, heading_cost_weight, obstacle_cost_weight, velocity_cost_weight", [
    ("test_1",0, 0, 0, 0, 0, 0.5, -0.7,  3.1, 1, 1),
    ("test_2",0, 0, 0, 0, 0, 0.85,    0,    3.1, 1, 1),
    ("test_3",0, 0, 0, 0, 0, 0.85,    0.75,    3.1, 1, 1),
    ("test_4",0, 0, 0, 0, 0,  0.5, 0.75,  3.1, 1, 1),
    ("test_5",0, 0, 0, 0, 0, -0.5, 0.4,  3.1, 1, 1),
    ("test_6",0, 0, 0, 0, 0, -0.5, 0.75,  3.1, 1, 1),
    ("test_7",0, 0, 0, 0, 0, -0.9, 0.75,  3.1, 1, 1),
    ("test_8",0, 0, 0, 0, 0, -1.2, 0.75, 3.1, 1, 1),
    ("test_9",0, 0, 0, 0, 0, -1.2, 0.0, 3.1, 1, 1),
    ("test_10",0, 0, 0, 0, 0, -1.2, -0.75, 3.1, 1, 1),
    ("test_11",0, 0, 0, 0, 0, -0.5, -0.7,  3.1, 1, 1),
    ("test_12",0, 0, 0, 0, 0, 0.5, -0.9,  3.1, 1, 1)
])  
def test_dwa_with_obs(plt, plt_name, init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y,
                    heading_cost_weight,obstacle_cost_weight,velocity_cost_weight):
    cm,obstacles = get_costmap("local_costmap_copy.txt")
    [x_traj,y_traj], [obs_x, obs_y],goal_state = go(init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y,
                                                heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,
                                                cm, obstacles)
    assert goal_state == True
    if goal_state:
        plt.plot(x_traj,y_traj, "-r")
    plt.plot(init_x, init_y, "xr")
    plt.plot(goal_x, goal_y, "xb")
    plt.plot(obs_x, obs_y, "ok")
    plot_arrow(init_x, init_y, init_theta)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.saveas = plt_name + ".png"
