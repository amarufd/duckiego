import numpy as np
import cv2

def find_centroid_lines(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_yellow = np.array([20,100,100])
  upper_yellow = np.array([30,255,255])

  mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
  kernel = np.ones((5,5), np.uint8)
  mask_erosion = cv2.erode(mask, kernel, iterations = 2)
  mask_dilation = cv2.erode(mask_erosion, kernel, iterations = 1)

  res = cv2.bitwise_and(hsv_img,hsv_img, mask = mask_dilation)
  res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
  res_bgr_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)

  th = cv2.adaptiveThreshold(res_bgr_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                    cv2.THRESH_BINARY_INV, 11, 2)

  contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  M = cv2.moments(contours[0])
  cX = int(M["m10"]/M["m00"])
  cY = int(M["m01"]/M["m00"])
  cv2.circle(th, (cX, cY), 50, (255, 255, 255), 0)
  print("X", cX)
  print("Y", cY)
  for c in contours:
    M = cv2.moments(c)

    if int(M["m00"]) != 0:
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      cv2.circle(img, (cX, cY), 50, (255, 255, 255), 0)
  return img

def main():
  img = cv2.imread("./lines/t005.png")
  centroids = find_centroid_lines(img)
  cv2.imshow("centroids", centroids)
  cv2.waitKey(0)
  cv2.destroyAllWindow()

if __name__ == "__main__":
  main()

