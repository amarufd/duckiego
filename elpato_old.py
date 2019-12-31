#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

####
from PIL import Image
import cv2
import math
from apriltag import Detector
import transformations as tf

#calculo steering
import LineRoadsDetection as LR

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckiebot-v0') #Ingresamos el ambiente
parser.add_argument('--map-name', default='udem1') #mapa
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()
total_reward = 0

def _draw_pose(overlay, camera_params, tag_size, pose, z_sign=1):

    opoints = np.array([
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1,  1, 0,
        -1, -1, -2*z_sign,
         1, -1, -2*z_sign,
         1,  1, -2*z_sign,
        -1,  1, -2*z_sign,
    ]).reshape(-1, 1, 3) * 0.5*tag_size

    edges = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4
    ]).reshape(-1, 2)
        
    fx, fy, cx, cy = camera_params

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
    ipoints = np.round(ipoints).astype(int)
    ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]
    for i, j in edges:
        cv2.line(overlay, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)

def global_pose(matrix,x_ob,y_ob,angle):
    #obtiene el angulo del tag con respecto al mapa
    q1 = math.atan2(y_ob,x_ob)
    # invierte el angulo del tag segun el plano del mapa
    angle = -angle
    # Calcula la distancia del robot al tag
    z = dist(matrix)
    # Calcula la distancia del tag al mapa
    d = math.sqrt(x_ob**2 + y_ob**2)
    # Calcula el angulo del robot c/r a q1
    q2 = angle2(q1,angle,tf.euler_from_matrix(matrix))
    R1 = tf.rotation_matrix(q1,[0,0,1])
    T1 = tf.translation_matrix([d,0,0])
    R2 = tf.rotation_matrix(q2,[0,0,1])
    T2 = tf.translation_matrix([z,0,0])
    result = R1.dot(T1.dot(R2.dot(T2.dot([0,0,0,1]))))
    
    return result

def angle2(q,angle,euler):
    return q-(angle-yaw(euler))

def l1(x,y):
    return math.sqrt(x**2,y**2)

def yaw(euler_angles):
    return euler_angles[2]

def dist(matrix):
    return np.linalg.norm([matrix[0][3],matrix[1][3],matrix[2][3]])
steering_angle=0#-math.pi
action=np.array([0,0])
i=0
while True:
    i=i+1

    #lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    #distance_to_road_center = lane_pose.dist
    #angle_from_straight_in_rads = lane_pose.angle_rad

    ###### Start changing the code here.
    # TODO: Decide how to calculate the speed and direction.

    k_p = 10
    k_d = 1
    
    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
    
    speed = 1.5 # TODO: You should overwrite this value
    
    # angle of the steering wheel, which corresponds to the angular velocity in rad/s
    #steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads # TODO: You should overwrite this value
    #print("stering real : \n"+str(steering))
    ###### No need to edit code below.
    
    #obs, reward, done, info = env.step([speed, steering_angle])
    
    obs, reward, done, info = env.step(action)

    ### line detector
    original = Image.fromarray(obs)
    original=np.ascontiguousarray(np.flip(original, axis=0))
    cv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    erode_it=1
    dilate_it=1
    height_y=150
    offset_y=150
    boundary=2/4

    #height_y=100
    #offset_y=40
    #boundary=2/4
    
    #Definición del rango de amarillos
    lower_yellow = np.array([18, 41, 133])
    upper_yellow = np.array([30, 255, 255])
    #Definición del rango de blancos 
    lower_white = np.array([0, 0, 75]) #V más grande menos gris
    upper_white = np.array([180, 5, 255])
    
    steering_angle = -LR.compute_steering(cv_img, erode_it, dilate_it, height_y, offset_y, boundary, lower_yellow, upper_yellow, lower_white, upper_white)*0.75
    action=np.array([0,0])

    if i%4==0:
        action = speed*np.array([+0.42*(math.cos(steering_angle)+math.sin(steering_angle)), -0.36*(math.cos(steering_angle)-math.sin(steering_angle))])
        #print(i)
    

    #print("stering CV : \n"+str(steering_angle))
    # line detector end
    print(steering_angle)
    #print(action)

    ### apriltags detector
    label = ""
    
    

    detector = Detector()
    gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    detections, dimg = detector.detect(gray, return_image=True)
    camera = [305.57, 308.83, 303.07, 231.88]

    pose = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    robot_pose = [0.0, 0.0]
    for detection in detections:
        #
        pose, e0, e1 = detector.detection_pose(detection, camera, 0.18 / 2 *0.585) #malo borrar división
        if not np.isnan(pose[0][0]):
            _draw_pose(cv_img,
                       camera,
                       0.18/ 2 *0.585,
                       pose)
            
        robot_pose = global_pose(pose, 2.08*0.585, 4.05*0.585, math.pi/2) #no es pi/2 es pi# está malaS
        
    label = 'detections = %d, dist = %.2f, pos = (%.2f, %.2f)' % (len(detections), pose[2][3], robot_pose[0], robot_pose[1])
    
    cv2.imshow('win', cv_img)
    cv2.waitKey(5)

    #if done:
    #   print('done!')
    #    env.reset()
    #    env.render()
    extra_label = pyglet.text.Label(
        font_name="Arial",
        font_size=14,
        x=5,
        y=600 - 19*2
    )
    env.render(mode="top_down")
    extra_label.text = label
    extra_label.draw()
    #Hasta acá
    total_reward += reward
    
    #print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    #env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        #print ('Final Reward = %.3f' % total_reward)
        #break
