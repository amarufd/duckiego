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

lower_yellow = np.array([20, 50, 50])
upper_yellow = np.array([25, 255, 255])
erode_it = 1
dilate_it = 1
####

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
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

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))


    ### duckie detector
    original = Image.fromarray(obs)
    cv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    # hsv image
    hsvImage = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # mask
    # Mascara para filtrar amarillo HSV
    mask = cv2.inRange(hsvImage, lower_yellow, upper_yellow) 
                
    kernel = np.ones((5,5),np.uint8) # Matriz para erosion y dilatacion
    erosion = cv2.erode(mask, kernel, iterations=erode_it)
    dilated = cv2.dilate(erosion, kernel, iterations=dilate_it)
    image_out = cv2.bitwise_and(cv_img, cv_img, mask=dilated) # Aplicar mascara

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Contornos externos, Aproximacion simple

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt) # x, y, width, height
        # Area minima del rectangulo = 1000 pixeles
        if w * h >= 500: 
            # Crear rectangulo en la imagen original
            cv2.rectangle(cv_img, (x,y), (x+w,y+h), (255,255,0), 2) 


    # Imagen modo espejo
    mirror_img = cv2.flip(cv_img, 1) 
    
    cv2.imshow('win', cv_img)
    cv2.waitKey(5)

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()