import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gplearn.genetic import SymbolicRegressor
import os
import pickle

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

def load_dataset(npz_file_path):
    """
        Loads a dataset from a .npz file.
        :param npz_file_path: The path to the .npz file
        :return: A tuple containing the images and labels arrays
        """
    data = np.load(npz_file_path)
    filepaths = data['filepaths']
    labels = data['labels']

    images = [
        img_to_array(load_img(filepath, color_mode='rgb', target_size=(100, 100))) / 255.0
        for filepath in filepaths]
    return np.array(images), np.array(labels)


def add_layers(model, filters, kernel_size=(3, 3), activation='relu', pool_size=(2, 2), input_shape=None):
    """
        Adds Conv2D and MaxPooling2D layers to the model.
        :param model: The model to add layers to
        :param filters: The number of output filters in the convolution
        :param kernel_size: The height and width of the 2D convolution window
        :param activation: Activation function to use
        :param pool_size: Factors by which to downscale
        :param input_shape: Optional shape tuple, only to be specified if it is the first layer in the model
        """

    if input_shape:
        model.add(Conv2D(filters, kernel_size, activation=activation, input_shape=input_shape))
    else:
        model.add(Conv2D(filters, kernel_size, activation=activation))
    model.add(MaxPooling2D(pool_size))

def laplacian(Z):
    """
    Return the Laplacian of matrix Z.
    :param Z: Input matrix
    :return: Laplacian of the input matrix
    """
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1) -
            4 * Z)

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
def mse(imageA, imageB):
    """
    Calculates the 'Mean Squared Error' between the two images.
    :param imageA: First image
    :param imageB: Second image
    :return: Mean Squared Error between the two images
    """
    # The 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images.
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def create_model(input_shape):
    """
        Creates a Sequential model with Conv2D, MaxPooling2D, Flatten, and Dense layers.
        :param input_shape: The shape of the input data (height, width, channels)
        :return: The created model
        """
    model = Sequential()
    add_layers(model, 32, input_shape=input_shape)
    add_layers(model, 64)
    add_layers(model, 128)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Set output_size to 2
    return model


npz_path = r'../gray_scott/gray_scott_data.npz'  # Ensure this file is in the correct location
image_folder = 'gray_scott'  # Ensure this folder is in the correct location



# Load the "stain-glass" image at 100*100 pixels
stain_glass_image = img_to_array(load_img('../Images_to_be_used/stain-glass.png', color_mode='rgb', target_size=(100, 100))) / 255.0

# Reshape the image to match the input shape of the model
stain_glass_image = np.expand_dims(stain_glass_image, axis=0)

# Print the shape of the image to confirm it's correct
print(f"Stain-glass image shape: {stain_glass_image.shape}")




images, labels = load_dataset(npz_path)

model = create_model((100, 100, 3))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(images, labels, epochs=20, batch_size=32, validation_split=0.2)

feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-3].output)
features = feature_extractor.predict(images)

regressor_params = {
    'population_size': 1000,
    'generations': 20,
    'stopping_criteria': 0.01,
    'p_crossover': 0.7,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.1,
    'max_samples': 0.9,
    'verbose': 1,
    'parsimony_coefficient': 0.01,
    'random_state': 0
}

for i in range(2):
    regressor = SymbolicRegressor(**regressor_params)
    regressor.fit(features, labels[:, i])
    print(f"Parameter {i + 1} Regression Model:\n{regressor._program}")
    with open(f'pkl/gray_scott_symbolic_regressor_{i + 1}.pkl', 'wb') as f:
        pickle.dump(regressor, f)

model.save('gray_scott_cnn_model.h5')


# Use the image as input to the model
model_output = model.predict(stain_glass_image)

# Print the output of the model
print(f"Model output: {model_output}")
# Load the trained model


# Extract features from the image
feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-3].output)
features = feature_extractor.predict(stain_glass_image)

# Use the features as input to the regressors to predict the parameters for the Gray-Scott model
param_gray_scott = []
for i in range(2):
    with open(f'pkl/gray_scott_symbolic_regressor_{i + 1}.pkl', 'rb') as f:
        regressor = pickle.load(f)
    param_gray_scott.append(regressor.predict(features)[0])

# Generate an image using the Gray-Scott model with the predicted parameters
# This assumes that you have a function `generate_gray_scott` that takes the parameters as input and saves the resulting image
image_path, _ = generate_gray_scott((100, 100), param_gray_scott[0], param_gray_scott[1], 0.16, 0.08, steps=30000)

# Display the original and generated images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axs[0].imshow(stain_glass_image[0])
axs[0].set_title('Original Image')
axs[0].axis('off')

# Generated image
generated_image = img_to_array(load_img(image_path, color_mode='rgb', target_size=(100, 100))) / 255.0


axs[1].imshow(generated_image)
axs[1].set_title('Generated Image')
axs[1].axis('off')
plt.show()

# Calculate the MSE between the original and generated images
error = mse(stain_glass_image[0], generated_image)
print(f"MSE: {error}")