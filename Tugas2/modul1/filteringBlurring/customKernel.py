import cv2
import numpy as np

img = cv2.imread('pemandangan.jpg')
kernel_sharpen = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])

sharpened = cv2.filter2D(img, -1, kernel_sharpen)

kernel_edge = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])

enhanced = cv2.filter2D(img, -1, kernel_edge)

kernel_emboss = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])

embossed = cv2.filter2D(img, -1, kernel_emboss)
