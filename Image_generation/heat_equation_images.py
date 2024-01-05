import numpy as np
import matplotlib.pyplot as plt
import os
import PIL.Image as Image

# Define the root directory of your project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def generate_heat_equation(k, output_dir, size=(100, 100), dpi=100):
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    X, Y = np.meshgrid(x, y)
    T = np.exp(-k * (X - 0.5) ** 2 - k * (Y - 0.5) ** 2)

    fig_size = (size[0] / dpi, size[1] / dpi)
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(T, extent=(0, 1, 0, 1), origin='lower', cmap='inferno')
    plt.axis('off')

    filename = f"heat_k_{k:.2f}.png"
    full_output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    filepath = os.path.join(full_output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return filepath, T

def resize_image(input_path, output_size=(400, 400)):
    with Image.open(input_path) as img:
        resized_img = img.resize(output_size, Image.Resampling.LANCZOS)
        resized_img.save(input_path)

def generate_and_save_images():
    directory = "heat_equation"
    if not os.path.exists(directory):
        os.makedirs(directory)

    thermal_conductivities = np.linspace(5, 100, 200)
    data_association = {}
    labels = []

    for k in thermal_conductivities:
        filepath, data = generate_heat_equation(k, directory, dpi=100)
        data_association[filepath] = data
        labels.append([k])

    np.savez(os.path.join(directory, 'data_association_heat.npz'), filepaths=list(data_association.keys()), datas=list(data_association.values()), labels=labels)

    print("Images and data association saved in 'heat_equation' directory.")

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            resize_image(file_path)

    print("All images have been resized.")