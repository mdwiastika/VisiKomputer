import cv2

img = cv2.imread('foto_noisy.jpg')
blurred = cv2.medianBlur(img, 5)
