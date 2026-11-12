import cv2

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)

# Blur dulu untuk mengurangi noise
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Kemudian Otsu
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Threshold Otsu: {ret}")
