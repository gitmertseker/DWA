import math
import numpy as np
from skimage.morphology import erosion,disk
import cv2
import time
from copy import deepcopy
from sklearn.preprocessing import normalize
from DWA import Path,Obstacle,RobotState,Robot,Costmap
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import seaborn as sns
import keyboard

sns.set()

#robot parameters
min_v = 0  # minimum translational velocity
max_v = 0.1  # maximum translational velocity
min_w = -math.pi/4    # minimum angular velocity
max_w = math.pi/4   # maximum angular velocity
max_a_v = 0.2  # maximum translational acceleration/deceleration
# max_a_w = 45 * math.pi /180  # maximum angular acceleration/deceleration
max_a_w = 90 * math.pi /180  # maximum angular acceleration/deceleration
max_dec_v = max_a_v
max_dec_w = max_a_w
delta_v = 0.1/2  # increment of translational velocity # window length / interval
delta_w = np.deg2rad(18/4)  # increment of angular velocity
# dt =  0.1     # time step
dt = 0.1
n =   30      # how many time intervals
# n = 40
obs_sensitivity = 0.5
# movement_path =

filename = "local_costmap_copy.txt"

init_x = 1
init_y = 0.8

init_x = 1.4
init_y = 1.2

init_theta = 0
# init_theta = np.pi/2
# init_theta = np.pi/8

init_v = 0
init_w =0

goal_x = 0
goal_y = 0.3

# goal_x = 0.4
# goal_y = 1.2

goal_x = 0.5
goal_y = 1.4


goal_x = 0.5
goal_y = 0.4


goal_x = 0.4
goal_y = 0.5
# goal_x = 2.15
# goal_y = 2.03

# goal_x = 4
# goal_y = 2

# goal_x = 10
# goal_y = 10

# goal_x = 6.5
# goal_y = 9.5

# goal_x = 3.8
# goal_y = 3.5

#movement_path main'in başından silindi



def plot_arrow(x, y, yaw, length=0.2, width=0.04):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            head_length=width, head_width=width)
    plt.plot(x, y)



def on_press(event,fig):
    if event.key == 'space':
        fig.set_visible(fig.get_visible())
    else:
        fig.set_visible(not fig.get_visible())


def main(filename,min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,obs_sensitivity,goal_region = 0.02):

    costmap = Costmap()
    cm = costmap.read_costmap(filename)
    #new_list = deepcopy(cm)
    cm = [[0 for i in row] for row in cm]   # empty costmap
    obstacles = costmap.find_obstacles(cm)
    cm_rev = costmap.cm_rev(cm)
    cm_rev2 = costmap.cm_norm(cm_rev)

    


    robot = Robot(cm_rev2,min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_dec_v,max_dec_w,delta_v,delta_w,dt,n,obs_sensitivity)
    state = RobotState(init_x,init_y,init_theta,init_v,init_w)

    # obstacles = [Obstacle(2,1), Obstacle(2,4), Obstacle(6,7),Obstacle(3,3),Obstacle(2,3), Obstacle(3,4),Obstacle(0.1,0), Obstacle(2,0)]
    # obstacles = [Obstacle(0.1,0), Obstacle(0.3,0)]
    # obstacles = [Obstacle(21,20), Obstacle(22,20), Obstacle(21,21),Obstacle(20,20),Obstacle(22,22)]
    # obstacles = [Obstacle(35,30),Obstacle(38,31),Obstacle(33,29),Obstacle(30,32),Obstacle(32,17)] #,Obstacle(34,20)
    # obstacles = [Obstacle(38,35)]
    # obstacles = [Obstacle(45,45),Obstacle(35,45),Obstacle(30,9),Obstacle(36,20)] #,Obstacle(45,33)
    
    # obstacles = [Obstacle(0,2),Obstacle(4,2),Obstacle(5,4),Obstacle(5,5),Obstacle(5,6),Obstacle(5,9),Obstacle(8,9),Obstacle(7,9),Obstacle(8,10),Obstacle(9,11),Obstacle(12,13)]
    # obstacles = [Obstacle(0,40),Obstacle(80,40),Obstacle(100,80),Obstacle(100,100),Obstacle(100,120),
    # Obstacle(100,180),Obstacle(160,180),Obstacle(140,180),Obstacle(160,200),Obstacle(180,220),Obstacle(240,260),
    # Obstacle(260,260),Obstacle(60,110),Obstacle(120,180),Obstacle(100,140),Obstacle(100,150),Obstacle(100,145)]
    
    # obstacles = [Obstacle(45,45),Obstacle(35,45),Obstacle(30,9)]

    # for i in movement_path:  #pathin hepsini tek seferde mi gönderecek yoksa tek tek mi??
    #     goal_x = i[0]
    #     goal_y = i[1] 

    # obs_x, obs_y = robot.obstacle_position(obstacles,state)
    obs_x, obs_y = robot.obs_pos_trial(obstacles)


    goal_Flag = False


    # if not goal_Flag:
    #     paths,opt_path = robot.calc_opt_traj(goal_x,goal_y,state,obstacles)   
    #     for cnt in range(len(opt_path.x)):
    #         print("x : "+ str(opt_path.x[cnt]) + "y :" + str(opt_path.y[cnt]) )

    #     for i in range(len(paths)):
            
    #         plt_x = paths[i].x
    #         plt_y = paths[i].y
    #         # plt.xlim(1.5,2.5)
    #         # plt.ylim(1.5,2.5)
    #         plt.xlim(0,8)
    #         plt.ylim(0,8)
    #         plt.plot(plt_x,plt_y,color = "blue",label = ("v = " + str(paths[i].v)+ " w = " + str(paths[i].w)))
    #     plt.plot(opt_path.x,opt_path.y,color = "red", label = "optimal path")
    #     plt.plot(state.x, state.y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    #     plt.plot(goal_x, goal_y, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
    #     for x,y in zip(obs_x,obs_y):
    #         plt.plot(x,y, marker="o", markersize=6, markeredgecolor="black", markerfacecolor="black")
    #     plt.legend()
    #     plt.show()


    #     # velocity commands
    #     opt_v = opt_path.v   
    #     opt_w = opt_path.w 
    #     print("Optimal velocities are: ({},{})".format((opt_v),(opt_w)))
    #     state.update_state(opt_v,opt_w,dt)

    #     dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)
    #     if dis_to_goal < goal_region:
            # goal_Flag = True


    show_animation = True
    plt_ctr = 0
    while not goal_Flag:
        
        paths,opt_path = robot.calc_opt_traj(goal_x,goal_y,state,obstacles)

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

            # plt.gcf().canvas.mpl_connect(
            #     'key_press_event',
                # lambda event: [plt.gcf.set_visible(plt.gcf.get_visible()) if event.key == 'space' else plt.gcf.set_visible(not plt.gcf.get_visible())])
            for path in paths:
                plt.plot(path.x, path.y, "-r")    
            plt.plot(opt_path.x, opt_path.y, "-g")

            plt.plot(x, y, "xr")
            plt.plot(goal_x, goal_y, "xb")
            plt.plot(obs_x, obs_y, "ok")
            plot_arrow(x, y, theta)
            plt.axis("equal")
            plt.grid(True)
            plt.xlim(0,2)
            plt.ylim(0,2)
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

# start_time = time.perf_counter()
print("Start!!")
main(filename,min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,obs_sensitivity)
# end_time = time.perf_counter()

# print("Run time = {} msec".format(1000*(end_time-start_time)))