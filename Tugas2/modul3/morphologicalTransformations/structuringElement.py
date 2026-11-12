import cv2
import numpy as np

# Kotak 5x5
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Elips 5x5
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Salib 5x5
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# Atau manual dengan numpy
kernel_manual = np.ones((3, 3), np.uint8)
