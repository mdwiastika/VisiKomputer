import cv2

img = cv2.imread('secangkir.webp', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges_clean = cv2.Canny(blurred, 100, 200)
