from pathlib import Path
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
img = cv2.imread('lingkaran.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    # Hitung luas dan keliling
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    print(f"Bentuk {i+1} -> Luas: {area} px², Keliling: {perimeter:.2f} px")

    # Bounding rectangle tegak (garis hijau)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Minimum enclosing circle (garis biru)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(img, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)

    # Titik tengah / centroid (lingkaran kuning terisi)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)

cv2.imshow('Properti Kontur', img)
cv2.waitKey(0)
cv2.destroyAllWindows()