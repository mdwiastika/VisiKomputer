import cv2
import numpy as np

img = cv2.imread('road.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi tepi
edges = cv2.Canny(gray, 50, 150)

# Standard Hough Transform
lines = cv2.HoughLines(edges, 
                       rho=1,           # Resolusi jarak (piksel)
                       theta=np.pi/180, # Resolusi sudut (radian)
                       threshold=100)   # Minimum votes

# Gambar garis
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
