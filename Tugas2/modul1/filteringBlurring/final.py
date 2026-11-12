import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('pemandangan.jpg')
h, w = img.shape[:2]
scale = 800 / max(h, w)
img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Tambahkan noise salt-and-pepper untuk testing
def add_salt_pepper(image, amount=0.02):
    noisy = image.copy()
    # Salt
    num_salt = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    num_pepper = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

noisy = add_salt_pepper(img_rgb)

# Apply berbagai blur
average = cv2.blur(noisy, (5, 5))
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
median = cv2.medianBlur(noisy, 5)
bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

# Tampilkan
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original')

axes[0, 1].imshow(noisy)
axes[0, 1].set_title('Noisy (Salt-Pepper)')

axes[0, 2].imshow(average)
axes[0, 2].set_title('Average Blur')

axes[1, 0].imshow(gaussian)
axes[1, 0].set_title('Gaussian Blur')

axes[1, 1].imshow(median)
axes[1, 1].set_title('Median Blur ✓')

axes[1, 2].imshow(bilateral)
axes[1, 2].set_title('Bilateral Filter')

plt.tight_layout()
plt.show()
