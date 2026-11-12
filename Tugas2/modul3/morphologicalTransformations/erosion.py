import cv2
import numpy as np

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)

# Buat kernel
kernel = np.ones((3, 3), np.uint8)

# Erosi
eroded = cv2.erode(img, kernel, iterations=1)

# Erosi lebih kuat (2 kali iterasi)
eroded_strong = cv2.erode(img, kernel, iterations=2)
