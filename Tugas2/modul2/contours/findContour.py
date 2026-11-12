import cv2
import numpy as np

img = cv2.imread('lingkaran.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE)

print(f"Jumlah kontur ditemukan: {len(contours)}")

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', img)
cv2.waitKey(0)
