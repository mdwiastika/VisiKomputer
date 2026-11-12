import cv2
import numpy as np

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((3, 3), np.uint8)

# Dilasi
dilated = cv2.dilate(img, kernel, iterations=1)

# Dilasi lebih kuat
dilated_strong = cv2.dilate(img, kernel, iterations=2)
