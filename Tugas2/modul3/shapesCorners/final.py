import cv2
import numpy as np

img = cv2.imread('ubin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Deteksi Garis ---
edges = cv2.Canny(blurred, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                        minLineLength=30, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
# --- Deteksi Lingkaran ---
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=100, param2=30,
                           minRadius=10, maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(img, (x, y), r, (255, 0, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

# --- Deteksi Corner ---
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
if corners is not None:
    for corner in np.int32(corners):
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 4, (0, 255, 255), -1)

cv2.imshow('Detection', img)
cv2.waitKey(0)
