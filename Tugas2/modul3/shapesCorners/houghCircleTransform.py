import cv2
import numpy as np

img = cv2.imread('coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur untuk kurangi noise
blurred = cv2.medianBlur(gray, 5)

# Deteksi lingkaran
circles = cv2.HoughCircles(blurred,
                           cv2.HOUGH_GRADIENT,
                           dp=1,           # Inverse ratio resolusi
                           minDist=50,     # Jarak minimum antar pusat
                           param1=100,     # Threshold Canny atas
                           param2=30,      # Threshold akumulator
                           minRadius=10,   # Radius minimum
                           maxRadius=100)  # Radius maksimum

# Gambar lingkaran
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, radius = circle
        # Gambar lingkaran
        cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
        # Gambar pusat
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
