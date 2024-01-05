import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def julia_set(height, width, c, x_min, x_max, y_min, y_max, max_iter=10000):
    x = np.linspace(x_min * np.pi, x_max * np.pi, width)  # Scale x by pi
    y = np.linspace(y_min * np.pi, y_max * np.pi, height)  # Scale y by pi
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    output = np.zeros(Z.shape, dtype=int)
    for i in range(max_iter):
        mask = np.abs(Z) < 2  # Change the threshold to 2
        Z[mask] = Z[mask] ** 2 + c
        output += mask

    return output

# Increase the resolution by increasing the height and width
height, width = 2000, 2000
# Define the point of interest
# Define the point of interest
x_point, y_point = 0.01201, 0

# Define the initial zoom level
zoom_level = 0.1

# Define the zoom factor
zoom_factor = 0.5

# Keep 'c' constant
c = -0.8 + 0.156j

# Generate and display Julia set images
plt.figure(figsize=(15, 15))

for i in range(10):
    # Adjust the zoom by changing x_min, x_max, y_min, y_max
    x_min, x_max = x_point - zoom_level, x_point + zoom_level
    y_min, y_max = y_point - zoom_level, y_point + zoom_level

    julia_image = julia_set(height, width, c, x_min, x_max, y_min, y_max, max_iter=100)
    plt.subplot(4, 3, i+1)
    plt.imshow(julia_image, cmap='inferno')
    plt.axis('off')
    plt.title(f'c = {c:.4f}, zoom = {zoom_level:.4f}')

    # Zoom in more for the next image
    zoom_level *= zoom_factor

plt.tight_layout()
plt.show()