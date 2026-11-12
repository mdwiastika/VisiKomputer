import cv2
import numpy as np

img = cv2.imread('secangkir.webp', cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_abs = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_display = cv2.convertScaleAbs(sobel_combined)
