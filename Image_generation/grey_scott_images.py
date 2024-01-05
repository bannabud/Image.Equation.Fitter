import numpy as np
import matplotlib.pyplot as plt
import os
import PIL.Image as Image

def laplacian(Z):
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1) -
            4 * Z)


def save_image(V, feed_rate, kill_rate, step=None):
    plt.imshow(V, cmap='inferno')
    plt.axis('off')
    step_suffix = f"_step_{step}" if step is not None else ""
    filename = f"gray_scott_fr_{feed_rate}_kr_{kill_rate}.png"
    filepath = os.path.join("gray_scott_renders", filename)
    if not os.path.exists("gray_scott_renders"):
        os.makedirs("gray_scott_renders")
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filepath, V

def generate_gray_scott(size, feed_rate, kill_rate, Du, Dv, steps=10000, save_steps=None):
    """
        Generates an image based on the Gray-Scott model.
        :type save_steps: object
        :param size: Size of the image (in pixels)
        :param feed_rate: Feed rate for the Gray-Scott model
        :param kill_rate: Kill rate for the Gray-Scott model
        :param Du: Diffusion rate of U
        :param Dv: Diffusion rate of V
        :param steps: Number of iterations for the model
        :param save_steps: Steps at which to save the image
        :return: File paths of the generated images
        """
    U, V = np.ones(size), np.zeros(size)
    mid = size[0] // 2
    r = 20
    U[mid-r:mid+r,mid-r:mid+r], V[mid-r:mid+r,mid-r:mid+r] = 0, 1

    saved_images = []

    for i in range(steps):
        Lu, Lv = laplacian(U), laplacian(V)
        U += Du * Lu - U*V**2 + feed_rate * (1 - U)
        V += Dv * Lv + U*V**2 - (feed_rate + kill_rate) * V

        if save_steps is not None and i in save_steps:
            filepath, _ = save_image(V, feed_rate, kill_rate, step=i)
            saved_images.append(filepath)

    print(np.unique(V))  # Print unique values in V

    return saved_images

def generate_and_save_images():
    image_directory = "gray_scott"
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    num_images = 300
    feed_rate_start, feed_rate_end = 0.030, 0.080
    kill_rate_start, kill_rate_end = 0.055, 0.065
    Du, Dv = 0.16, 0.08
    size = (100, 100)

    data_collection = {}
    labels = []

    for i in range(num_images):
        feed_rate = np.random.uniform(feed_rate_start, feed_rate_end)
        kill_rate = np.random.uniform(kill_rate_start, kill_rate_end)
        filepath, data = generate_gray_scott(size, feed_rate, kill_rate, Du, Dv, steps=10000)
        data_collection[filepath] = data
        labels.append((feed_rate, kill_rate))

    np.savez(f'{image_directory}/gray_scott_data.npz', filepaths=list(data_collection.keys()), datas=list(data_collection.values()), labels=labels)

    print("All images and labels saved.")