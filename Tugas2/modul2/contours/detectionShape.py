import cv2

img = cv2.imread('lingkaran.jpg')


h, w = img.shape[:2]
total_area = h * w

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def detect_shape(contour):
    epsilon = 0.015 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return "Segitiga"
    elif vertices == 4:
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect_ratio = bw / float(bh)
        return "Persegi" if 0.95 <= aspect_ratio <= 1.05 else "Persegi Panjang"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    else:
        return "Lingkaran"

for contour in contours:
    area = cv2.contourArea(contour)
    
    if area < 500 or area > (total_area * 0.95):
        continue

    shape_name = detect_shape(contour)
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(img, shape_name, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Deteksi Nama Bentuk', img)
cv2.waitKey(0)
cv2.destroyAllWindows()