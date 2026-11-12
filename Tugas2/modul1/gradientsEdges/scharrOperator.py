import cv2

img = cv2.imread('secangkir.webp', cv2.IMREAD_GRAYSCALE)

scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharr_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
