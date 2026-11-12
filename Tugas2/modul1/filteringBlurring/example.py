import os
import cv2
import numpy as np

img = cv2.imread('pemandangan.jpg')
if img is None:
    print("Error: Gambar tidak ditemukan! Periksa kembali path 'pemandangan.jpg'")
    exit(1)

kernel_blur = np.ones((3, 3), np.float32) / 9
kernel_sharp = np.array([[0, -1, 0],         
                         [-1, 5, -1],
                         [0, -1, 0]])

blurred = cv2.filter2D(img, -1, kernel_blur)
sharpened = cv2.filter2D(img, -1, kernel_sharp)
