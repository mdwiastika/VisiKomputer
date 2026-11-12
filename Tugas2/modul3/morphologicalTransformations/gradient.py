import cv2
import numpy as np

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)

# Gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
