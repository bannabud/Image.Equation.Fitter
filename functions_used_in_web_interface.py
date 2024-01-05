import os
import pickle
import numpy as np
import tensorflow as tf
from keras.src.utils import array_to_img
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import rc

from Image_generation.heat_equation_images import generate_heat_equation
from Image_generation.grey_scott_images import generate_gray_scott
from Image_generation.Lattice_images import generate_spacetime_effect

# Define the root directory of your project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

rc('text', usetex=False)
rc('font', size=16)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def preprocess_image(image_path, target_size=(400, 400)):
    img = load_img(image_path, color_mode='rgb', target_size=target_size)
    return img_to_array(img)[np.newaxis, ...] / 255.0


def extract_features(model_path, image):
    full_model_path = os.path.join(ROOT_DIR, model_path)
    model = tf.keras.models.load_model(full_model_path)
    feature_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    features = feature_model.predict(image)
    tf.keras.backend.clear_session()
    return features


def save_image(V, feed_rate, kill_rate, step=None):
    plt.imshow(V, cmap='inferno')
    plt.axis('off')
    step_suffix = f"_step_{step}" if step is not None else ""
    filename = f"gray_scott_fr_{feed_rate}_kr_{kill_rate}.png"
    filepath = os.path.join(".../gray_scott_renders", filename)
    if not os.path.exists("gray_scott_renders"):
        os.makedirs("gray_scott_renders")
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filepath, V


def load_dataset(npz_file_path):
    directory = os.path.dirname(npz_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Now you can open your file
    if os.path.exists(npz_file_path):
        data = np.load(npz_file_path)
        filepaths = data['filepaths']
        labels = data['labels']
        return filepaths, labels
    else:
        raise FileNotFoundError(f"No file or directory found at {npz_file_path}")

def run_model(image_path):
    try:
        processed_image = preprocess_image(image_path)
        features_mandelbrot = extract_features('h5/cnn_mandelbrot_model.h5', processed_image)
        print("Mandelbrot model has successfully run and generated an image.")

        features_heat_eq = extract_features('h5/cnn_heat_model.h5', processed_image)
        print("Heat equation model has successfully run and generated an image.")

        features_gray_scott = extract_features('h5/gray_scott_cnn_model.h5',
                                               preprocess_image(image_path, target_size=(100, 100)))
        print("Gray-Scott model has successfully run and generated an image.")

        features_lattice = extract_features('h5/cnn_lattice_model.h5', processed_image)
        print("Lattice model has successfully run and generated an image.")
        full_pickle_path = os.path.join(ROOT_DIR, 'pkl', 'symbolic_regressor_mandelbrot.pkl')
        with open(full_pickle_path, 'rb') as f:
            symbolic_regressor_mandelbrot = pickle.load(f)
        full_pickle_path = os.path.join(ROOT_DIR, 'pkl', 'symbolic_regressor_heat.pkl')
        with open(full_pickle_path, 'rb') as f:
            symbolic_regressor_heat_eq = pickle.load(f)
        full_pickle_path = os.path.join(ROOT_DIR, 'pkl', 'gray_scott_symbolic_regressor_1.pkl')
        with open(full_pickle_path, 'rb') as f:
            symbolic_regressor_gray_scott_1 = pickle.load(f)
        full_pickle_path = os.path.join(ROOT_DIR, 'pkl', 'gray_scott_symbolic_regressor_2.pkl')
        with open(full_pickle_path, 'rb') as f:
            symbolic_regressor_gray_scott_2 = pickle.load(f)
        full_pickle_path = os.path.join(ROOT_DIR, 'pkl', 'symbolic_regressor_lattice.pkl')
        with open(full_pickle_path, 'rb') as f:
            symbolic_regressor_lattice = pickle.load(f)

        param_mandelbrot = symbolic_regressor_mandelbrot.predict(features_mandelbrot)
        param_heat_eq = symbolic_regressor_heat_eq.predict(features_heat_eq)
        param_gray_scott_1 = symbolic_regressor_gray_scott_1.predict(features_gray_scott)[0]
        param_gray_scott_2 = symbolic_regressor_gray_scott_2.predict(features_gray_scott)[0]
        param_lattice = symbolic_regressor_lattice.predict(features_lattice)

        full_npz_path = os.path.join(ROOT_DIR, 'mandelbrot_images', 'data_association_mandelbrot.npz')
        images, labels = load_dataset(full_npz_path)
        differences = np.abs(labels - param_mandelbrot)
        index = np.argmin(differences)
        image_mandelbrot = images[index]

        # Generate only one image based on the fitted parameters
        filepath_heat, image_heat_eq = generate_heat_equation(param_heat_eq[0], 'heat_equation_images')

        # Generate Gray-Scott model image
        image_paths = generate_gray_scott((100, 100), param_gray_scott_1, param_gray_scott_2, 0.16, 0.08, steps=20000, save_steps=[19999])

        # Load and upscale the Gray-Scott model image
        image_gray_scott_small = img_to_array(load_img(image_paths[-1], color_mode='rgb', target_size=(100, 100))) / 255.0
        image_gray_scott = array_to_img(image_gray_scott_small).resize((400, 400))

        image_lattice = generate_spacetime_effect(400,1000,param_lattice[0])
        fig, axs = plt.subplots(1, 5, figsize=(25, 7))

        axs[0].imshow(processed_image[0])
        axs[0].set_title('Original Image', fontsize=25)
        axs[0].axis('off')

        from matplotlib.image import imread
        full_image_path = os.path.join(ROOT_DIR, 'mandelbrot_images', 'mandelbrot_time_299.93333333333334_frame_8998.png')
        if os.path.exists(full_image_path):
            image_mandelbrot = imread(full_image_path)
            axs[1].imshow(image_mandelbrot)
            axs[1].set_title('Mandelbrot Representation', fontsize=25)
            axs[1].axis('off')
        else:
            raise FileNotFoundError(f"No file or directory found at {full_image_path}")


        axs[2].imshow(image_heat_eq, cmap='viridis')
        axs[2].set_title('Heat Equation Representation', fontsize=25)
        axs[2].axis('off')

        axs[3].imshow(image_gray_scott, cmap='viridis')
        axs[3].set_title('Gray-Scott Representation', fontsize=25)
        axs[3].axis('off')

        axs[4].imshow(image_lattice)
        axs[4].set_title('Lattice Representation', fontsize=25)
        axs[4].axis('off')

        plt.tight_layout()
        output_image_path = os.path.join(ROOT_DIR, 'output.png')
        plt.savefig(output_image_path)
        plt.close()

        return output_image_path
    except Exception as e:
        print(f"An error occurred in run_model: {e}")
        return None