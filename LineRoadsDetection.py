import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

class DETECTION:
    def __init__(self, frame, height_y, offset_y, boundary):
        self.frame=frame
        self.height, self.width, _= frame.shape
        self.height_y=height_y
        self.offset_y=offset_y
        self.boundary=boundary

    def interest_region(self, mask):
        region_a_recortar = [[0,int(self.height/2+self.height_y+self.offset_y)],[0,int(self.height)],[int(self.width),int(self.height)],[int(self.width),int(self.height/2+self.height_y+self.offset_y)]]
        poligono_a_recortar = np.array([region_a_recortar],dtype=np.int32)
        cv2.fillPoly(mask,poligono_a_recortar,0)
        region_a_recortar = [[0,int(self.height/2-self.height_y+self.offset_y)],[0,0],[int(self.width),0],[int(self.width),int(self.height/2-self.height_y+self.offset_y)]]
        poligono_a_recortar = np.array([region_a_recortar],dtype=np.int32)
        cv2.fillPoly(mask,poligono_a_recortar,0)
        return mask

    def detect_line_segments(self, edges):

        rho = 1
        angle = np.pi / 180
        min_threshold = 20
        line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold, 
                                        np.array([]), minLineLength=8, maxLineGap=10)

        return line_segments

    def average_slope_intercept(self, line_segments):
        lane_lines = []
        if line_segments is None:
            return lane_lines

        left_fit = []
        left_weights=[]
        right_fit = []
        right_weights=[]

        left_region_boundary = self.width * (1 - self.boundary)  # la linea de la izquierda está por dentro de la variable boundary por la izquierda
        right_region_boundary = self.width * self.boundary # la linea de la derecha está por dentro de la variable boundary por la derecha

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    #ignora lineas verticales
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                length=np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                        if abs(slope) >0.09: #se corrige el problema de las líneas horizontales (amarilla segmentada)
                            left_weights.append((length))
                        else:
                            left_weights.append((1)) #cuando es horizontal el peso es 1
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))
                        if abs(slope) >0.09:
                            right_weights.append((length))
                        else:
                            right_weights.append((1))

        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit,weights=left_weights, axis=0)
            lane_lines.append(make_points(left_fit_average, self.height, self.width, self.height_y, self.offset_y))

        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit,weights=right_weights, axis=0)
            lane_lines.append(make_points(right_fit_average, self.height, self.width, self.height_y, self.offset_y))

        return lane_lines
    

    def display_lines(self, lines, line_color=(0, 255, 0), line_width=2):
        line_image = np.zeros_like(self.frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        line_image = cv2.addWeighted(self.frame, 0.8, line_image, 1, 1)
        return line_image

def make_points(line, height, width, height_y, offset_y):
    slope, intercept = line
    y1=int(height/2+height_y+offset_y) #abajo
    y2=int(height/2-height_y+offset_y) #arriba

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def compute_steering(cv_img, erode_it, dilate_it, height_y, offset_y, boundary, lower_yellow, upper_yellow, lower_white, upper_white):
    height, width, _= cv_img.shape
    #Se transforma la imagen a HSV
    hsv_img=cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    #Se realiza una segmentación por color creando dos mascaras que definen el
    #blanco y amarillo
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    mask_white =  cv2.inRange(hsv_img, lower_white, upper_white)

    #Se unen ambas máscaras, así se tiene una segmentación de blancos y amarillos
    mask=cv2.bitwise_or(mask_white,mask_yellow)
    #Inicializamos la detección
    lds=DETECTION(cv_img, height_y, offset_y, boundary)
    #Asignamos un espacio de interés de la máscara
    mask=lds.interest_region(mask)
    #Realizamos erosión y dilatación
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=erode_it)
    dilated = cv2.dilate(erosion, kernel, iterations=dilate_it)
    #obtenemos una imágen ajustada a la máscara
    image_out = cv2.bitwise_and(cv_img, cv_img, mask=dilated) # Aplicar mascara

    #Se detectan los bordes con el método Canny de OPENCV
    edges=cv2.Canny(mask,200,400)

    #Se usa el procedimiento line_segments para detectar las líneas en la imágen
    line_segments=lds.detect_line_segments(edges)
    #Se calcula una línea que es el resultado de un promedio ponderado de todas las líneas
    #detectadas
    lane_lines = lds.average_slope_intercept(line_segments)
    #Como output se muestra la imágen
    lane_lines_image = lds.display_lines(lane_lines)

    if len(lane_lines)==0:
        x_offset=0
        y_offset = int(height/2-height_y+offset_y)
    elif len(lane_lines)==1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height/2-height_y+offset_y)
    elif len(lane_lines)==2:
        _, _, left_x2, _=lane_lines[0][0]
        _, _, right_x2, _=lane_lines[1][0]
        mid=int(width/2)
        x_offset=(left_x2+right_x2)/2-mid
        y_offset=int(height/2-height_y+offset_y)


    steering_angle = math.atan(x_offset / y_offset) #radianes
    if x_offset==0:
        steering_angle=+math.pi*1/12
    #if abs(steering_angle)>90*math.pi/180:
    #    steering_angle=steering_angle/abs(steering_angle)*10*math.pi/180

    #cv2.imshow("original", cv_img)
    cv2.imshow("maskara", mask)
    cv2.imshow("image salida",image_out)
    cv2.imshow("lane lines", lane_lines_image)
    #cv2.imwrite( "lines/imagen.jpg", lane_lines_image)
    #cv2.waitKey(20000)

    return steering_angle


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('-datadir', type=str, required=True)
    args=parser.parse_args()
    str_datadir=args.datadir
    cv_img = cv2.imread(str_datadir)
    #cv_img = cv2.imread("./lines/t007.png")

    erode_it=1
    dilate_it=1
    height_y=100
    offset_y=100
    boundary=2/3
    
    #Definición del rango de amarillos
    lower_yellow = np.array([18, 41, 133])
    upper_yellow = np.array([30, 255, 255])
    #Definición del rango de blancos 
    lower_white = np.array([0, 0, 75]) #V más grande menos gris
    upper_white = np.array([180, 5, 255])

    steering_angle = compute_steering(cv_img, erode_it, dilate_it, height_y, offset_y, boundary, lower_yellow, upper_yellow, lower_white, upper_white)
    print(steering_angle)