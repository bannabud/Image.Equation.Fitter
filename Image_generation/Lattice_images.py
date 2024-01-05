import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom

from PIL import Image

def generate_spacetime_effect(size, num_layers, distortion_strength):
    image = np.zeros((size, size))
    center = (size // 2, size // 2)

    for layer in range(num_layers):
        num_lines = 500 + 10 * layer
        angles = np.linspace(0, 2 * np.pi, num_lines)

        for angle in angles:
            x = np.cos(angle) * np.linspace(0, size // 2, 200) + center[0]
            y = np.sin(angle) * np.linspace(0, size // 2, 200) + center[1]

            x = np.clip(x.astype(int), 0, size - 1)
            y = np.clip(y.astype(int), 0, size - 1)

            # Adjust the intensity increment based on the layer or total number of lines
            intensity_increment = 1 / (num_lines + layer * 10)  # Example adjustment
            image[x, y] += intensity_increment

    # Normalize the image with a more nuanced approach
    max_intensity = np.percentile(image, 99)  # Use the 99th percentile as the scaling factor
    if max_intensity > 0:  # Avoid division by zero
        image = np.clip(image / max_intensity, 0, 1)

        # Make sure to pass the distortion_strength to the apply_distortion function
    return apply_distortion(image, center, distortion_strength)

def apply_distortion(image, center, distortion_strength):
    size = image.shape[0]
    distorted_image = np.zeros_like(image)

    for i in range(size):
        for j in range(size):
            offset_x, offset_y = calculate_distortion(i, j, center, distortion_strength)  # Use the distortion_strength

            src_x = (i + int(offset_x)) % size
            src_y = (j + int(offset_y)) % size

            distorted_image[i, j] = image[src_x, src_y]

    return distorted_image

def calculate_distortion(x, y, center, distortion_strength):
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    angle = np.arctan2(y - center[1], x - center[0])

    # Utilize the passed distortion_strength in the calculation
    frequency = 30  # Adjust frequency as needed
    offset_x = distortion_strength * np.sin(distance / frequency) * np.cos(angle)
    offset_y = distortion_strength * np.sin(distance / frequency) * np.sin(angle)

    return offset_x, offset_y


def visualize_patterns(pattern, distortion_strength, save_dir):
    filename = f"spacetime_effect_{distortion_strength}.png"
    plt.imsave(os.path.join(save_dir, filename), pattern, cmap='nipy_spectral')
import shutil

def resize_image(input_path, output_size=(400, 400)):
    with Image.open(input_path) as img:
        resized_img = img.resize(output_size, Image.Resampling.BILINEAR)
        resized_img.save(input_path)


def main():
    size = 400  # Size of the grid
    num_layers = 1000  # Number of layers
    save_dir = "../Lattice_images"  # Specify your directory here

    # Delete the directory if it exists
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # Create the directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the data association dictionary and labels list
    data_association = {}
    labels = []

    for i in range(1, 200):  # Loop to create 5 images with increasing distortion strength
        distortion_strength = 40 * i  # Incrementing distortion strength
        pattern = generate_spacetime_effect(size, num_layers, distortion_strength)  # Now correctly passing 3 arguments
        filename = visualize_patterns(pattern, distortion_strength, save_dir)

        # Store the file path and data in the data association dictionary
        data_association[filename] = pattern

        # Store the distortion strength in the labels list
        labels.append([distortion_strength])



    # Save the association using numpy in .npz format
    np.savez(os.path.join(save_dir, 'data_association_lattice.npz'), filepaths=list(data_association.keys()), datas=list(data_association.values()), labels=labels)

    print("Images and data association saved in 'Lattice_images' directory.")

if __name__ == "__main__":
    main()



