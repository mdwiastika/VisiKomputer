from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)


# Resize
h, w = img.shape[:2]
scale = 600 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)))

# Gunakan Adaptive Threshold Gaussian agar kebal terhadap bayangan miring
binary = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=15,
    C=8
)

# Structuring element
kernel = np.ones((2, 2), np.uint8)

# Operasi Morfologi
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# Plot hasil
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(binary, cmap='gray'); axes[0, 0].set_title('Original Binary (Teks Saja)')
axes[0, 1].imshow(eroded, cmap='gray'); axes[0, 1].set_title('Eroded (Teks Menipis)')
axes[0, 2].imshow(dilated, cmap='gray'); axes[0, 2].set_title('Dilated (Teks Menebal)')
axes[1, 0].imshow(opened, cmap='gray'); axes[1, 0].set_title('Opening')
axes[1, 1].imshow(closed, cmap='gray'); axes[1, 1].set_title('Closing')
axes[1, 2].imshow(gradient, cmap='gray'); axes[1, 2].set_title('Gradient')

plt.tight_layout()
plt.show()