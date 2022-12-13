import math
import numpy as np
from skimage.morphology import erosion,disk
import cv2
from copy import deepcopy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt




class Path:
    def __init__(self,v,w):
        self.x = None
        self.y = None
        self.theta = None
        self.v = v
        self.w = w


class Obstacle:
    def __init__(self,x,y):
        self.x = x
        self.y = y



class Costmap:

    def read_costmap(self,filename):

        with open(filename, "r") as f:
            x=[]
            
            for line in f.readlines():
                x.append(line[:-1])
            y=[]
            line=""
            for i in range(0,len(x),1):
                line=line+x[i]
                if x[i][-1]==']':
                    line=line.replace('   ',' ')
                    line=line.replace('  ',' ')
                    line=line.replace('[ ','[')
                    line=line.replace(' ]',']')
                    b=line[1:-1]
                    l=b.split(' ')
                    y.append(list(map(int,l) ))
                    line=""
        return y


    def find_obstacles(self,cm):
        obstacles = []
        for z in range(len(cm[0])):
            for i in range(len(cm[0])):
                if cm[i][z] < 0.03:
                    obs_temp = Obstacle(i,z)
                    obstacles.append(obs_temp)
        return obstacles


    def cm_rev(self,cm):
        cm_rev = deepcopy(cm)
        for i in range (0,len(cm[0])):
            for z in range(0,len(cm[1])):
                cm_rev[i][z] = 100-cm[i][z]
        return cm_rev


    def cm_norm(self,cm_rev):


        return [[i/100 for i in row] for row in cm_rev]


class RobotState:
    def __init__(self,init_x,init_y,init_theta,init_v,init_w):

        self.x = init_x
        self.y = init_y
        self.theta = init_theta

        self.v = init_v
        self.w = init_w

        self.traj_x = [init_x]
        self.traj_y = [init_y]
        self.traj_theta = [init_theta]

    def update_state(self,v,w,dt):

        self.v = v
        self.w = w
        self.dt = dt

        next_x = (self.v * math.cos(self.theta) * self.dt) + self.x
        next_y = (self.v * math.sin(self.theta) * self.dt) + self.y
        next_theta = (self.w * self.dt) + self.theta

        self.traj_x.append(next_x)
        self.traj_y.append(next_y)
        self.traj_theta.append(next_theta)

        self.x = next_x
        self.y = next_y
        self.theta = next_theta

        return self.x, self.y, self.theta


    def traj(self):
        return self.traj_x,self.traj_y

class Robot:
    def __init__(self,costmap,min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_dec_v,max_dec_w,delta_v,delta_w,dt,n,obs_sensitivity):
        #robot parameters

        self.min_v = min_v       # minimum translational velocity
        self.max_v = max_v       # maximum translational velocity
        self.min_w = min_w       # minimum angular velocity
        self.max_w = max_w       # maximum angular velocity
        self.max_a_v = max_a_v     # maximum translational acceleration/deceleration
        self.max_a_w = max_a_w     # maximum angular acceleration/deceleration
        self.max_dec_v = max_dec_v
        self.max_dec_w = max_dec_w
        self.delta_v = delta_v      #increment of velocity
        self.delta_w = delta_w     #increment of angular velocity
        self.dt = dt          # time step
        self.n = n           #how many time intervals


        self.costmap = costmap
        self.obs_sensitivity = obs_sensitivity
        self.v_count = 5
        # self.w_count = 6
        self.w_count = 5
        # self.heading_cost_weight = 0.1
        # self.heading_cost_weight = 3.1
        self.heading_cost_weight = 10
        # self.obstacle_cost_weight = 0.8
        self.obstacle_cost_weight = 5
        self.velocity_cost_weight = 0.4

        self.traj_paths = []
        self.traj_opt = []


    # state = RobotState(0.0,0.0,0.0)


    def angle_correction(self,angle):

        if angle > math.pi:
            while angle > math.pi:
                angle -=  2 * math.pi
        elif angle < -math.pi:
            while angle < -math.pi:
                angle += 2 * math.pi

        return angle



    def calc_dw(self,state):

        self.cur_v = state.v
        self.cur_w = state.w

        Vs = [self.min_v, self.max_v, self.min_w, self.max_w]  #maximum velocity area
        Vd = [(self.cur_v-self.max_a_v*self.dt), (self.cur_v+self.max_a_v*self.dt), (self.cur_w-self.max_a_w*self.dt), (self.cur_w+self.max_a_w*self.dt)]   #velocities robot can generate until next time step
        min_v = max(Vs[0], Vd[0])
        max_v = min(Vs[1], Vd[1])
        min_w = max(Vs[2], Vd[2])
        max_w = min(Vs[3], Vd[3])
        dw = [min_v, max_v, min_w, max_w]

        return dw




    def predict_state(self,v,w,x,y,theta,dt,n):

        next_x_s = []
        next_y_s = []
        next_theta_s = []

        for i in range(n):
            temp_x = (v * math.cos(theta) * dt) + x
            temp_y = (v * math.sin(theta) * dt) + y
            temp_theta = w * dt + theta

            next_x_s.append(temp_x)
            next_y_s.append(temp_y)
            next_theta_s.append(temp_theta)

            x = temp_x
            y = temp_y
            theta = temp_theta

        return next_x_s, next_y_s, next_theta_s


    def calc_opt_traj(self,goal_x,goal_y,state,obstacles,goal_region):

        paths = self.make_path(state)
        opt_path = self.eval_path(paths,goal_x,goal_y,obstacles,state,goal_region)
        
        self.traj_opt.append(opt_path)

        return paths, opt_path


    def make_path(self,state):

        dw = self.calc_dw(state)
        self.min_v_dw = dw[0]
        self.max_v_dw = dw[1]
        self.min_w_dw = dw[2]
        self.max_w_dw = dw[3]

        paths = []

        for w in np.linspace(self.min_w_dw,self.max_w_dw,self.w_count):
            for v in np.linspace(self.min_v_dw,self.max_v_dw,self.v_count):
                if not (v == 0 and w == 0):
                    path = Path(v,w)
                    next_x, next_y, next_theta = self.predict_state(v,w,state.x,state.y,state.theta,self.dt,self.n)

                    path.x = next_x
                    path.y = next_y
                    path.theta = next_theta
                    
                    paths.append(path)
                # print("path number :" + str(len(paths)-1)+" ,linear speed :" + str(v) + " ,angular speed :" +str(w))

        self.traj_paths.append(paths)

        return paths


    def normalize_1d(self,list):
        norm = np.linalg.norm(list)
        list = list/norm
        return list



    def eval_path(self,paths,goal_x,goal_y,obstacles,state,goal_region):
        
        score_headings_temp = []
        score_velocities_temp = []
        score_obstacles_temp = []
        score_cost_temp = []
        score_obstacles = []

        for path in paths:

            score_headings_temp.append(self.calc_heading(path,goal_x,goal_y,state,goal_region))
            score_velocities_temp.append(self.calc_velocity(path))
            # score_obstacles_temp.append(self.calc_clearance(path,obstacles,state))
            score_obstacles.append(self.calc_clearance(path,state)) #iceride normalize ediliyor !!


        #normalization
        # print("****Before normalization heading cost****: {}".format(score_headings))
        score_headings = [h/math.pi for h in score_headings_temp]  
        # print("****After normalization heading cost****: {}".format(score_headings)) 

        # print("****Before normalization velocity cost****: {}".format(score_velocities))
        score_velocities = [h/self.max_v_dw for h in score_velocities_temp]
        # print("****After normalization velocity cost****: {}".format(score_velocities))

        # print("****Before normalization obstacle cost****: {}".format(score_obstacles))
        # score_obstacles = [h/self.score_obstacle_max for h in score_obstacles_temp]


        # if all(i > 1 for i in score_obstacles):
        #     raise("No path found, surrounded by obstacles")

        # print("****After normalization obstacle cost****: {}".format(score_obstacles))

        # score_headings = self.normalize_1d(score_headings)
        # score_velocities = self.normalize_1d(score_velocities)
        # score_obstacles = self.normalize_1d(score_obstacles)

        score = 0

        for k in range(len(paths)):
            temp_score = 0

            temp_score = (self.heading_cost_weight*score_headings[k]) + (self.obstacle_cost_weight*score_obstacles[k]) + (self.velocity_cost_weight*score_velocities[k])

            if temp_score > score:
                if paths[k].v == 0 and paths[k].w == 0:
                    print("= ******0 hız isteği*****")
                    continue
                opt_path = paths[k]
                score = temp_score
                # print(str(k)+ ". path is optimal for now, score :"+str(score))
        try:
            return opt_path
        except:
            raise("Can not calculate optimal path!")



    # def eval_path(self,paths,goal_x,goal_y,obstacles,state): #minimize eden versiyon
        
    #     score_headings = []
    #     score_velocities = []
    #     score_obstacles_temp = []
    #     score_cost_temp = []
    #     score_obstacles = []

    #     for path in paths:

    #         score_headings.append(self.calc_heading(path,goal_x,goal_y))
    #         score_velocities.append(self.calc_velocity(path))
    #         # score_obstacles_temp.append(self.calc_clearance(path,obstacles,state))
    #         score_obstacles.append(self.obs_check(path,obstacles,state)) #iceride normalize ediliyor !!


    #     #normalization
    #     # print("****Before normalization heading cost****: {}".format(score_headings))
    #     # score_headings = [h/math.pi for h in score_headings_temp]  
    #     # print("****After normalization heading cost****: {}".format(score_headings)) 

    #     # print("****Before normalization velocity cost****: {}".format(score_velocities))
    #     # score_velocities = [h/self.max_v_dw for h in score_velocities_temp]
    #     # print("****After normalization velocity cost****: {}".format(score_velocities))

    #     # print("****Before normalization obstacle cost****: {}".format(score_obstacles))
    #     # score_obstacles = [h/self.score_obstacle_max for h in score_obstacles_temp]


    #     # if all(i > 1 for i in score_obstacles):
    #     #     raise("No path found, surrounded by obstacles")

    #     # print("****After normalization obstacle cost****: {}".format(score_obstacles))

    #     # score_headings = self.normalize_1d(score_headings)
    #     # score_velocities = self.normalize_1d(score_velocities)
    #     # score_obstacles = self.normalize_1d(score_obstacles)

    #     score = float("inf")

    #     for k in range(len(paths)):
    #         temp_score = 0

    #         temp_score = (self.heading_cost_weight*score_headings[k]) + (self.obstacle_cost_weight*score_obstacles[k]) + (self.velocity_cost_weight*score_velocities[k])

    #         if temp_score < score:
    #             if paths[k].v == 0 and paths[k].w == 0:
    #                 print("= ******0 hız isteği*****")
    #                 continue
    #             opt_path = paths[k]
    #             score = temp_score
    #             # print(str(k)+ ". path is optimal for now, score :"+str(score))
    #     try:
    #         return opt_path
    #     except:
    #         raise("Can not calculate optimal path!")


    # def calc_heading(self,path,goal_x,goal_y):
        
    #     last_x = path.x[0]
    #     last_y = path.y[0]
    #     last_theta = path.theta[0]

    #     angle_to_goal = math.atan2((goal_y-last_y),(goal_x-last_x))
    #     score_angle = angle_to_goal-last_theta
    #     score_angle = abs(self.angle_correction(score_angle))
    #     score_angle = math.pi - score_angle


    #     return score_angle





    # def calc_heading(self,path,goal_x,goal_y): #gercekci olanı !!!!
        
    #     next_x = path.x[1]
    #     next_y = path.y[1]
    #     next_theta = path.theta[1]

    #     tfd_v = path.v / self.max_dec_v
    #     tfd_w = path.w / self.max_dec_w

    #     if tfd_v > self.dt:
    #         tfd_v = self.dt
    #     if tfd_w > self.dt:
    #         tfd_w = self.dt


    #     new_x = next_x - math.cos(next_theta)*(self.max_dec_v*(tfd_v**2))/2
    #     new_y = next_y - math.sin(next_theta)*(self.max_dec_v*(tfd_v**2))/2
    #     new_theta = next_theta - (self.max_dec_w*(tfd_w**2))/2


    #     angle_to_goal = math.atan2((goal_y-new_y),(goal_x-new_x))
    #     score_angle = angle_to_goal-new_theta
    #     score_angle = abs(self.angle_correction(score_angle))
    #     score_angle = math.pi - score_angle

    #     return score_angle




    def calc_heading(self,path,goal_x,goal_y,state,goal_region):
        
        dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)

        a = 2
        # if len(path.x)-1 > dis_to_goal > 50*goal_region:
        #     a = -1
        # else:
        #     a = 2

        last_x = path.x[a]
        last_y = path.y[a]
        last_theta = path.theta[a]

        angle_to_goal = math.atan2((goal_y-last_y),(goal_x-last_x))
        score_angle = angle_to_goal-last_theta
        # score_angle = abs(self.angle_correction(score_angle))

        cost = abs(math.atan2(math.sin(score_angle), math.cos(score_angle)))

        # score_angle = math.pi - score_angle
        score_cost = math.pi - cost

        return score_cost


    # def calc_heading(self,path,goal_x,goal_y): #minimize eden versiyon
        
    #     last_x = path.x[-1]
    #     last_y = path.y[-1]
    #     last_theta = path.theta[-1]

    #     angle_to_goal = math.atan2((goal_y-last_y),(goal_x-last_x))
    #     score_angle = angle_to_goal-last_theta
    #     score_angle = abs(self.angle_correction(score_angle))
    #     # score_angle = math.pi - score_angle

    #     return score_angle    


    def calc_velocity(self,path):

        score_velocity = path.v

        return score_velocity



    # def calc_velocity(self,path): #minimize eden versiyon

    #     score_velocity = path.v

    #     return 1/score_velocity



    def obstacle_position(self,obstacles,state):

        obs_x = []
        obs_y = []

        for obs in obstacles:
            # (20,20) initial robot position, resolution = 5 cm/pixel

            obs_x_temp = state.x + 0.05*(obs.x-20)
            obs_y_temp = state.y + 0.05*(obs.y-20)
            obs_x.append(obs_x_temp)
            obs_y.append(obs_y_temp)

        return obs_x,obs_y


    def obs_pos_trial(self,obstacles):

        obs_x = []
        obs_y = []

        for obs in obstacles:
            # (20,20) initial robot position, resolution = 5 cm/pixel

            obs_x_temp = 0.05*obs.x
            obs_y_temp = 0.05*obs.y
            obs_x.append(obs_x_temp)
            obs_y.append(obs_y_temp)

        return obs_x,obs_y




    def calculateSlope(self,x1,x2,y1,y2):
        (px,py) = (abs(x2-x1),abs(y2-y1))
        try: #slope should between 0 - 1
            if py > px:
                slope = px/py
            if py == 0 and px == 0:
                slope = 0
            else:
                slope = py/px
        except ZeroDivisionError:
            slope = 0
        # print("slope =: {}".format(slope))
        return slope


    def distance(self,x1,x2,y1,y2):
        px = (float(x1)-float(x2))**2
        py = (float(y1)-float(y2))**2
        return (px+py)**(0.5)


    def meter2pixel(self,x,state,resolution=0.05):

        if x > state.x:       
            x_pixel = (math.ceil((x-state.x)/resolution)) + 20
        else:
            x_pixel = (math.floor((x-state.x)/resolution)) + 20

        return x_pixel


#v3

    # def obs_check(self,path,obstacles,state):
        
    #     obs_x,obs_y = self.obs_pos_trial(obstacles)
    #     dx = path.x-obs_x
    #     dy = path.y-obs_y
    #     r = math.hypot(dx,dy)

    #     if np.array(r <= 0.05).any():
    #         return float("Inf")

    #     min_r = np.min(r)
    #     return 1/min_r


    def calc_clearance(self,path,state):  ## ****** YANLIS CALISIYOR !!! ****** 

        # AYNI PİKSELE GELEN NOKTALAR İCİN COST VE PATH UZUNLUGU HESAPLANMASINDA HATA VAR !!

        # self.score_obstacle_max = 96
        # self.obs_score = []
        # self.cost_score = []
        self.temp_cost = 0
        self.temp_obs = 0
        temp_stp = 0
        cost_sum = []
        cost_temp = 0

        for a in range(len(path.x)-1):
            x1 = path.x[a]
            x2 = path.x[a+1]

            y1 = path.y[a]
            y2 = path.y[a+1]
        

            x1 = self.meter2pixel(x1,state)
            x2 = self.meter2pixel(x2,state)
            y1 = self.meter2pixel(y1,state)
            y2 = self.meter2pixel(y2,state)
            pix_ctr = max(abs(x2-x1),abs(y2-y1))
            if pix_ctr == 0:
                pix_ctr = 1
            temp_stp = pix_ctr + temp_stp

        # if temp_stp == 0:
        #     temp_stp = 1

        for a in range(len(path.x)-1):
            x1 = path.x[a]
            x2 = path.x[a+1]

            y1 = path.y[a]
            y2 = path.y[a+1]        

            x1 = self.meter2pixel(x1,state)
            x2 = self.meter2pixel(x2,state)
            y1 = self.meter2pixel(y1,state)
            y2 = self.meter2pixel(y2,state)
            slope = self.calculateSlope(x1,x2,y1,y2)
            sign_x = np.sign(x2-x1)
            sign_y = np.sign(y2-y1)
            stp =  max(abs(x2-x1),abs(y2-y1))
            
            if stp == 0:
                if x1<40 and y1<40:
                    if a > 0:
                        x3 = self.meter2pixel(path.x[a-1],state)
                        y3 = self.meter2pixel(path.y[a-1],state)
                        if max(abs(x2-x1),abs(y2-y1),abs(y3-y2),abs(x3-x2)) == 0:
                            if not (a == len(path.x)-2):
                                continue
                            else:
                                return cost_temp/temp_stp
                    if self.costmap[x1][y1] <0.03:
                        return cost_temp/temp_stp
                    else:
                        cost_temp = cost_temp + self.costmap[x1][y1]
            else:
                for i in range (1,stp+1):
                    if abs(x2-x1) > abs(y2-y1):
                        xx = x1 + sign_x*i
                        yy = y1 + math.floor(sign_y*i*slope)
                        if xx<40 and yy<40:
                            if self.costmap[xx][yy] <0.03:
                                return cost_temp/temp_stp
                            else:
                                cost_temp = cost_temp + self.costmap[xx][yy]
                    else:
                        yy = y1 + sign_y*i
                        xx = x1 + math.floor(sign_x*i*(slope))               
                        if xx<40 and yy<40:
                            if self.costmap[xx][yy] <0.03:
                                return cost_temp/temp_stp
                            else: 
                                cost_temp = cost_temp + self.costmap[xx][yy]

        cost_norm = cost_temp/temp_stp

        return cost_norm





#v2
    # def calc_clearance(self,path,obstacles,state):
        
    #     resolution = 0.05
    #     resolution = 1
    #     self.score_obstacle_max = math.sqrt(2) #20 piksel*5cm(resolution)/100(cm2m conversion)
    #     self.score_obstacle_max = 30
    #     # self.score_obstacle_max = 10
    #     obs_x,obs_y = self.obstacle_position(obstacles,state)
    #     sum = 0
    #     for a in range(len(path.x)-1):
    #         x1 = path.x[a]
    #         x2 = path.x[a+1]

    #         y1 = path.y[a]
    #         y2 = path.y[a+1]

    #         slope = self.calculateSlope(x1,x2,y1,y2)

    #         sign_x = np.sign(x2-x1)
    #         sign_y = np.sign(y2-y1)
    #         xx = x1 + sign_x*resolution
    #         yy = y1 + sign_y*resolution

    #         stp =  math.floor((max(abs(x2-x1),abs(y2-y1)))/resolution)
    #         if stp == 0:
    #             for b in range(len(obs_x)):
    #                 if ((abs(x1 - obs_x[b])<resolution) and (abs(y1 - obs_y[b])<resolution)): #obstacle olma durumu
    #                     print("*** CrossObstacle *** v:{}, w: {}".format(path.v,path.w))
    #                     for c in range(a,0,-1):
    #                         sum = sum + self.distance(path.x[c-1],path.x[c],path.y[c-1],path.y[c-1])
    #                     sum = sum + self.distance(path.x[a],obs_x[b],path.y[a],obs_y[b])
    #                     return sum
    #         else:                
    #             for i in range (1,stp+1):
    #                 i = i*resolution
    #                 if abs(x2-x1) > abs(y2-y1):
    #                     xx = x1 + sign_x*i
    #                     yy = y1 + (sign_y*i*slope)
    #                     for z in range(-1,2):
    #                         z = z*0.05
    #                         for b in range(len(obs_x)):
    #                             if (abs(xx+z - obs_x[b]) <self.obs_sensitivity and abs(yy - obs_y[b])< self.obs_sensitivity): #obstacle olma durumu
    #                                 print("*** CrossObstacle *** v:{}, w: {}".format(path.v,path.w))
    #                                 for c in range(a,0,-1):
    #                                     sum = sum + self.distance(path.x[c-1],path.x[c],path.y[c-1],path.y[c-1])
    #                                 sum = sum + self.distance(path.x[a],xx+z,path.y[a],yy)
    #                                 return sum
    #                 else:
    #                     yy = y1 + sign_y*i
    #                     xx = x1 + (sign_x*i*(slope))               
    #                     for z in range(-1,2):
    #                         z = z*resolution
    #                         for b in range(len(obs_x)):
    #                             if (abs(xx - obs_x[b]) <self.obs_sensitivity and abs(yy+z - obs_y[b])< self.obs_sensitivity): #obstacle olma durumu
    #                                 print("*** CrossObstacle *** v:{}, w: {}".format(path.v,path.w))
    #                                 for c in range(a,0,-1):
    #                                     sum = sum + self.distance(path.x[c-1],path.x[c],path.y[c-1],path.y[c-1])
    #                                 sum = sum + self.distance(path.x[a],xx,path.y[a],yy+z)
    #                                 return sum
    #     return self.score_obstacle_max  



#v1
    # def calc_clearance(self,path,obstacles,state):

    #     self.score_obstacle_max = math.sqrt(2) #20 piksel*5cm(resolution)/100(cm2m conversion)
    #     score_obstacle = self.score_obstacle_max
    #     temp_dist = 0

    #     for i in range(len(path.x)):
    #         for obs in obstacles:
                

    #             obs_x = state.x + 0.05*(obs.x-20)

    #             obs_y = state.y + 0.05*(obs.y-20)

    #             temp_dist = math.sqrt((path.x[i]-obs_x)**2+(path.y[i]-obs_y)**2)
    #             # if temp_dist > 20:
    #             #     print("**** distance 20den büyük ****")
    #             if temp_dist < score_obstacle:
    #                 score_obstacle = temp_dist
    #     return score_obstacle

       




