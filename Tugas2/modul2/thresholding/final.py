import cv2
import matplotlib.pyplot as plt

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)

_, simple = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

adaptive_mean = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

adaptive_gaussian = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

blur = cv2.GaussianBlur(img, (5, 5), 0)
_, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(simple, cmap='gray')
axes[0, 1].set_title('Simple (T=127)')

axes[0, 2].imshow(adaptive_mean, cmap='gray')
axes[0, 2].set_title('Adaptive Mean')

axes[1, 0].imshow(adaptive_gaussian, cmap='gray')
axes[1, 0].set_title('Adaptive Gaussian')

axes[1, 1].imshow(otsu, cmap='gray')
axes[1, 1].set_title("Otsu's")

# Histogram
axes[1, 2].hist(img.ravel(), 256, [0, 256])
axes[1, 2].set_title('Histogram')

plt.tight_layout()
plt.show()
