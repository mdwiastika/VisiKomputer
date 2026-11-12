import cv2
import matplotlib.pyplot as plt

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)

plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
