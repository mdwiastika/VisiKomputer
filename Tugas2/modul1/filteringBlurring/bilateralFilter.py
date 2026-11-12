import cv2

img = cv2.imread('pemandangan.jpg')
blurred = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
