import cv2
import numpy as np

# Baca gambar
img = cv2.imread('lingkaran.jpg')
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur dan threshold
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

# Temukan kontur
contours, hierarchy = cv2.findContours(binary, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

# Analisis setiap kontur
for i, contour in enumerate(contours):
    # Filter kontur kecil (noise)
    area = cv2.contourArea(contour)
    if area < 100:
        continue
    
    # Gambar kontur
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
    
    # Centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
    
    # Label
    cv2.putText(img, f"A={int(area)}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

cv2.imshow('Contour Analysis', img)
cv2.waitKey(0)
