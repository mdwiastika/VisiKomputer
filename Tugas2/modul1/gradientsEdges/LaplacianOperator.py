import cv2

img = cv2.imread('secangkir.webp', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_display = cv2.convertScaleAbs(laplacian)
