import math
import numpy as np
import time
from copy import deepcopy
from DWA import Path,Obstacle,RobotState,Robot,Costmap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

#robot parameters
min_v = 0  # minimum translational velocity
max_v = 0.1  # maximum translational velocity
min_w = -math.pi/4    # minimum angular velocity
max_w = math.pi/4   # maximum angular velocity
max_a_v = 0.2  # maximum translational acceleration/deceleration
max_a_w = 90 * math.pi /180  # maximum angular acceleration/deceleration
max_dec_v = max_a_v
max_dec_w = max_a_w
delta_v = 0.1/2  # increment of translational velocity # window length / interval
delta_w = np.deg2rad(18/4)  # increment of angular velocity
dt =  0.1     # time step
n =   30      # how many time intervals


filename = "local_costmap_copy.txt"

init_x = 0 
init_y = 0
init_theta = 0


init_v = 0
init_w =0


goal_x = -0 
goal_y = -0.5 #h_c = 6

# goal_x = 0
# goal_y = 0.7  #h_c = 15



def plot_arrow(x, y, yaw, length=0.2, width=0.04):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            head_length=width, head_width=width)
    plt.plot(x, y)



def main(filename,min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,
    heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,goal_region = 0.02):

    orig_px=30
    costmap = Costmap()
    cm=np.zeros((orig_px*2,orig_px*2))
    cm[orig_px-20:orig_px+20,orig_px-20:orig_px+20]=costmap.read_costmap(filename)
    cm_rev = costmap.cm_rev(cm)
    cm_rev2 = costmap.cm_norm(cm_rev)
    obstacles = costmap.find_obstacles(cm_rev2)

    robot = Robot(cm_rev2,min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_dec_v,max_dec_w,delta_v,delta_w,dt,n,
                    heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,orig_px)
    state = RobotState(init_x,init_y,init_theta,init_v,init_w)


    # obs_x, obs_y = robot.obstacle_position(obstacles,state)
    obs_x, obs_y = robot.obs_pos_trial(obstacles)


    goal_Flag = False


    show_animation = True
    plt_ctr = 0
    while not goal_Flag:
        
        paths,opt_path = robot.calc_opt_traj(goal_x,goal_y,state,obstacles,goal_region)

        # velocity commands
        opt_v = opt_path.v   
        opt_w = opt_path.w 
        print("Optimal velocities are: ({},{})".format((opt_v),(opt_w)))
        x,y,theta = state.update_state(opt_v,opt_w,dt)


        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            for path in paths:
                plt.plot(path.x, path.y, "-r")    
            plt.plot(opt_path.x, opt_path.y, "-g")

            plt.plot(x, y, "xr")
            plt.plot(goal_x, goal_y, "xb")
            plt.plot(obs_x, obs_y, "ok")
            plot_arrow(x, y, theta)
            #plt.axis("equal")
            plt.grid(True)
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            if plt_ctr == 0:
                plt.pause(1)
            # plt.pause(0.001)
            # plt.pause(0.5)
            
            while not plt.waitforbuttonpress(): pass
            plt_ctr += 1


        dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)
        if dis_to_goal < goal_region:
            print("Goal!!")
            goal_Flag = True

    print("Done")
    if show_animation:
        x_traj,y_traj = state.traj()
        plt.plot(x_traj,y_traj, "-r")
        plt.pause(0.001)
    plt.show()


# heading_cost_weight = 3.1
heading_cost_weight = 6
# heading_cost_weight = 15
obstacle_cost_weight = 1
velocity_cost_weight = 1
goal_region = 0.03

print("Start!!")
# start_time = time.perf_counter()
main(filename,min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,
        heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,goal_region)
# end_time = time.perf_counter()

# print("Run time = {} msec".format(1000*(end_time-start_time)))