#!/usr/bin/env python
# manual

import sys
import argparse
import pyglet
import cv2
import numpy as np
import gym
import gym_duckietown
import math
import transformations as tf
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from PIL import Image
from apriltag import Detector

class DeteccionLineas():
    def __init__(self, frame, altura_y, correccion_y, limites):
        self.frame=frame
        self.altura, self.ancho, _= frame.shape
        self.altura_y=altura_y
        self.correccion_y=correccion_y
        self.limite=limites

    # Regio que donde se evaluan las lineas que se seguiran
    def regionEvaluar(self, mascara):
        region_a_recortar = [[0,int(self.altura/2+self.altura_y+self.correccion_y)],[0,int(self.altura)],[int(self.ancho),int(self.altura)],[int(self.ancho),int(self.altura/2+self.altura_y+self.correccion_y)]]
        poligono_a_recortar = np.array([region_a_recortar],dtype=np.int32)
        cv2.fillPoly(mascara,poligono_a_recortar,0)
        region_a_recortar = [[0,int(self.altura/2-self.altura_y+self.correccion_y)],[0,0],[int(self.ancho),0],[int(self.ancho),int(self.altura/2-self.altura_y+self.correccion_y)]]
        poligono_a_recortar = np.array([region_a_recortar],dtype=np.int32)
        cv2.fillPoly(mascara,poligono_a_recortar,0)
        return mascara

    # Deteccion de los segmentos que ayudaran a la conduccion del pato por el centro
    def detectarSegmentosLineas(self, esquinas):
        rho = 1
        angulo = np.pi / 180
        lim_menor = 20
        segmentos_lineas = cv2.HoughLinesP(esquinas, rho, angulo, lim_menor, 
                                        np.array([]), minLineLength=8, maxLineGap=10)
        return segmentos_lineas

    # Calcula el promedio de la interseccion de las rectas para el manejo del pato
    def promedioInterseccionRectas(self, segmentos_lineas):
        lineas_seguir = []
        if segmentos_lineas is None:
            return lineas_seguir

        ajuste_izq = []
        ancho_izq=[]
        ajuste_der = []
        ancho_der=[]

        limite_izq = self.ancho * (1 - self.limite)  # la linea de la izquierda está por dentro de la variable limites por la izquierda
        limite_der = self.ancho * self.limite # la linea de la derecha está por dentro de la variable limites por la derecha

        for line_segment in segmentos_lineas:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    #ignora lineas verticales
                    continue
                ajuste = np.polyfit((x1, x2), (y1, y2), 1)
                inclinacion = ajuste[0]
                interseccion = ajuste[1]
                length=np.sqrt((y2-y1)**2+(x2-x1)**2)
                if inclinacion < 0:
                    if x1 < limite_izq and x2 < limite_izq:
                        ajuste_izq.append((inclinacion, interseccion))
                        if abs(inclinacion) >0.09: #se corrige el problema de las líneas horizontales (amarilla segmentada)
                            ancho_izq.append((length))
                        else:
                            ancho_izq.append((1)) #cuando es horizontal el peso es 1
                else:
                    if x1 > limite_der and x2 > limite_der:
                        ajuste_der.append((inclinacion, interseccion))
                        if abs(inclinacion) >0.09:
                            ancho_der.append((length))
                        else:
                            ancho_der.append((1))

        if len(ajuste_izq) > 0:
            ajuste_izq_average = np.average(ajuste_izq,weights=ancho_izq, axis=0)
            lineas_seguir.append(puntos(ajuste_izq_average, self.altura, self.ancho, self.altura_y, self.correccion_y))

        if len(ajuste_der) > 0:
            ajuste_der_average = np.average(ajuste_der,weights=ancho_der, axis=0)
            lineas_seguir.append(puntos(ajuste_der_average, self.altura, self.ancho, self.altura_y, self.correccion_y))

        return lineas_seguir
    

    def muestraLineas(self, lineas, linea_color=(0, 255, 0), linea_ancho=2):
        lineas_imagen = np.zeros_like(self.frame)
        if lineas is not None:
            for linea in lineas:
                for x1, y1, x2, y2 in linea:
                    cv2.line(lineas_imagen, (x1, y1), (x2, y2), linea_color, linea_ancho)
        lineas_imagen = cv2.addWeighted(self.frame, 0.8, lineas_imagen, 1, 1)
        return lineas_imagen


parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown-udem1-v0') #Ingresamos el ambiente
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

def puntos(line, altura, ancho, altura_y, correccion_y):
    inclinacion, interseccion = line
    y1=int(altura/2+altura_y+correccion_y) #abajo
    y2=int(altura/2-altura_y+correccion_y) #arriba

    # encuadrando las coordenadas con el cuadro
    x1 = max(-ancho, min(2 * ancho, int((y1 - interseccion) / inclinacion)))
    x2 = max(-ancho, min(2 * ancho, int((y2 - interseccion) / inclinacion)))
    return [[x1, y1, x2, y2]]

def calcula_direccion(cv_img, erode_it, dilate_it, altura_y, correccion_y, limites, amarillo_claro, amarillo_oscuro, blanco_claro, blanco_oscuro):
    
    altura, ancho, _= cv_img.shape
    imagen_hsv=cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # Segmentacion para los colores de borde de la calle (blanco y amarillo) y se unen  las mascaras
    mascara_amarilla = cv2.inRange(imagen_hsv, amarillo_claro, amarillo_oscuro)
    mascara_blanca =  cv2.inRange(imagen_hsv, blanco_claro, blanco_oscuro)
    mascara=cv2.bitwise_or(mascara_blanca,mascara_amarilla)

    # Deteccion de lineas para la conduccion del pato
    lineas_detectadas=DeteccionLineas(cv_img, altura_y, correccion_y, limites)
    mascara=lineas_detectadas.regionEvaluar(mascara)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mascara, kernel, iterations=erode_it)
    dilated = cv2.dilate(erosion, kernel, iterations=dilate_it)
    
    esquinas=cv2.Canny(mascara,200,400)

    segmentos_lineas=lineas_detectadas.detectarSegmentosLineas(esquinas)
    lineas_calle = lineas_detectadas.promedioInterseccionRectas(segmentos_lineas)

    if len(lineas_calle)==0:
        x_correccion=0
        y_correccion = int(altura/2-altura_y+correccion_y)
    elif len(lineas_calle)==1:
        x1, _, x2, _ = lineas_calle[0][0]
        x_correccion = x2 - x1
        y_correccion = int(altura/2-altura_y+correccion_y)
    elif len(lineas_calle)==2:
        _, _, left_x2, _=lineas_calle[0][0]
        _, _, right_x2, _=lineas_calle[1][0]
        mid=int(ancho/2)
        x_correccion=(left_x2+right_x2)/2-mid
        y_correccion=int(altura/2-altura_y+correccion_y)

    direccion_angulo = math.atan(x_correccion / y_correccion)
    if x_correccion==0:
        direccion_angulo=+math.pi*1/12

    
    lineas_detectadas = lineas_detectadas.muestraLineas(lineas_calle)
    imagen_salida = cv2.bitwise_and(cv_img, cv_img, mask=dilated)
    cv2.imshow("mascara", mascara)
    cv2.imshow("image salida",imagen_salida)
    cv2.imshow("lane lineas", lineas_detectadas)

    return direccion_angulo


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

    esquinas = np.array([
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
    for i, j in esquinas:
        cv2.line(overlay, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)

def global_pose(matrix,x_ob,y_ob,angulo):
    #obtiene el angulo del tag con respecto al mapa
    q1 = math.atan2(y_ob,x_ob)
    # invierte el angulo del tag segun el plano del mapa
    angulo = -angulo
    # Calcula la distancia del robot al tag
    z = dist(matrix)
    # Calcula la distancia del tag al mapa
    d = math.sqrt(x_ob**2 + y_ob**2)
    # Calcula el angulo del robot c/r a q1
    q2 = angulo2(q1,angulo,tf.euler_from_matrix(matrix))
    R1 = tf.rotation_matrix(q1,[0,0,1])
    T1 = tf.translation_matrix([d,0,0])
    R2 = tf.rotation_matrix(q2,[0,0,1])
    T2 = tf.translation_matrix([z,0,0])
    result = R1.dot(T1.dot(R2.dot(T2.dot([0,0,0,1]))))
    
    return result

def angulo2(q,angulo,euler):
    return q-(angulo-yaw(euler))

def l1(x,y):
    return math.sqrt(x**2,y**2)

def yaw(euler_angulos):
    return euler_angulos[2]

def dist(matrix):
    return np.linalg.norm([matrix[0][3],matrix[1][3],matrix[2][3]])
direccion_angulo=-math.pi
while True:


    pose_del_pato = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distancia_de_manejo = pose_del_pato.dist
    angulo_recto = pose_del_pato.angle_rad

    k_p = 10
    k_d = 1
    speed = 0.2 

    # angulo del volante, que corresponde a la velocidad angular en rad/s
    direccion = k_p*distancia_de_manejo + k_d*angulo_recto
    print("stering real : \n"+str(direccion))

    obs, reward, done, info = env.step([speed, direccion_angulo])

    ### detectar lineas
    original = Image.fromarray(obs)
    cv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    imagen_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    erode_it=1
    dilate_it=1
    altura_y=75
    correccion_y=150
    limites=6/7
    
    #Definición del rango de amarillos
    amarillo_claro = np.array([18, 41, 133])
    amarillo_oscuro = np.array([30, 255, 255])
    #Definición del rango de blancos 
    blanco_claro = np.array([0, 0, 75]) #V más grande menos gris
    blanco_oscuro = np.array([180, 5, 255])

    direccion_angulo = -calcula_direccion(cv_img, erode_it, dilate_it, altura_y, correccion_y, limites, amarillo_claro, amarillo_oscuro, blanco_claro, blanco_oscuro)*3/4
    print("stering CV : \n"+str(direccion_angulo))
    # line detector end


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