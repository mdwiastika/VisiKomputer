import cv2
import numpy as np

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)

# Closing (dilasi lalu erosi)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
