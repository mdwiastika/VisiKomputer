import cv2

img = cv2.imread('kertas.jpg', cv2.IMREAD_GRAYSCALE)

adaptive_mean = cv2.adaptiveThreshold(
    img,
    255,                           # maxValue
    cv2.ADAPTIVE_THRESH_MEAN_C,    # metode: rata-rata
    cv2.THRESH_BINARY,             # tipe output
    11,                            # blockSize (harus ganjil)
    2                              # C (konstanta dikurangi dari mean)
)

# Adaptive threshold - Gaussian
adaptive_gaussian = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # metode: Gaussian-weighted
    cv2.THRESH_BINARY,
    11,
    2
)
