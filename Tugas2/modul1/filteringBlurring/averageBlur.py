import cv2

img = cv2.imread('pemandangan.jpg')

blurred = cv2.blur(img, (5, 5))

import numpy as np
kernel = np.ones((5, 5), np.float32) / 25
blurred = cv2.filter2D(img, -1, kernel)
