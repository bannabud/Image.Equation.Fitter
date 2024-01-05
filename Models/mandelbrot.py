import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gplearn.genetic import SymbolicRegressor
import os
import pickle

def create_and_compile_model(input_shape, output_size):
    """
     Creates and compiles a Sequential model with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
     :param input_shape: The shape of the input data (height, width, channels)
     :param output_size: The size of the output layer
     :return: The compiled model
     """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),  # Additional Conv2D and MaxPooling2D layers
        Conv2D(256, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),  # Increased number of neurons
        Dropout(0.5),  # Dropout layer
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout layer
        Dense(output_size, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_dataset(npz_file_path):
    """
      Loads a dataset from a .npz file.
      :param npz_file_path: The path to the .npz file
      :return: A tuple containing the images and labels arrays
      """
    data = np.load(npz_file_path)
    filepaths = data['filepaths']
    labels = data['labels']

    images = [img_to_array(load_img(filepath, color_mode='rgb', target_size=(400, 400))) / 255.0 for filepath in filepaths]
    return np.array(images), np.array(labels)

def save_model(model, filename):
    """
     Saves a model to a .h5 file.
     :param model: The model to save
     :param filename: The path to the .h5 file
     """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    model.save(filename)

npz_path = '../mandelbrot_images/data_association_mandelbrot.npz'
images, labels = load_dataset(npz_path)

model = create_and_compile_model((400, 400, 3), 1)
model.fit(images, labels.ravel(), epochs=20, batch_size=32, validation_split=0.2)

features = Model(inputs=model.inputs, outputs=model.layers[-3].output).predict(images)

regressor = SymbolicRegressor(population_size=1000, generations=20, stopping_criteria=0.01, p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=0.9, verbose=1, parsimony_coefficient=0.01, random_state=0)
regressor.fit(features, labels.ravel())

save_model(model, '../h5/cnn_mandelbrot_model.h5')

with open('../pkl/symbolic_regressor_mandelbrot.pkl', 'wb') as f:
    pickle.dump(regressor, f)