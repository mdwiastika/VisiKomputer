import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('secangkir.webp', cv2.IMREAD_GRAYSCALE)

# Blur untuk kurangi noise
blurred = cv2.GaussianBlur(img, (3, 3), 0)

# Sobel
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

# Scharr
scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
scharr = cv2.magnitude(scharr_x, scharr_y)
scharr = cv2.convertScaleAbs(scharr)

# Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Canny
canny = cv2.Canny(blurred, 100, 200)

# Tampilkan
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(sobel, cmap='gray')
axes[0, 1].set_title('Sobel')

axes[0, 2].imshow(scharr, cmap='gray')
axes[0, 2].set_title('Scharr')

axes[1, 0].imshow(laplacian, cmap='gray')
axes[1, 0].set_title('Laplacian')

axes[1, 1].imshow(canny, cmap='gray')
axes[1, 1].set_title('Canny ✓')

axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
