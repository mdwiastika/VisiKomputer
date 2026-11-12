import cv2

img = cv2.imread('pemandangan.jpg')

blurred = cv2.GaussianBlur(img, (5, 5), 0)

blurred_custom = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
