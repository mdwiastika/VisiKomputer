import cv2

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)

thresh_value = 127
ret, thresh_binary = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

ret, thresh_binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh_tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh_tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
