import cv2
from PIL import Image
import os
import numpy as np

# Define the root directory of your project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the output directory
output_dir = os.path.join(ROOT_DIR, '..', 'mandelbrot_images')
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the path to the video file
video_path = os.path.join(ROOT_DIR,'mandelbrot.mp4')

# Open the video file
video = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# Calculate the frequency of frames to be extracted
frame_freq = total_frames // 500

# Initialize the data association dictionary and labels list
data_association = {}
labels = []

# Loop over all frames in the video
for frame_count in range(total_frames):
    # Read the next frame
    success, frame = video.read()
    # If the frame was not read successfully, break the loop
    if not success:
        break

    # If the current frame number is a multiple of the frame frequency
    if frame_count % frame_freq == 0:
        # Calculate the time point corresponding to the current frame
        time_point = frame_count / video.get(cv2.CAP_PROP_FPS)
        # Define the output path for the frame image
        output_path = os.path.join(output_dir, f'mandelbrot_time_{time_point}_frame_{frame_count}.png')
        # Save the frame as an image
        cv2.imwrite(output_path, frame)

        # Open the saved image
        with Image.open(output_path) as img:
            # Get the size of the image
            width, height = img.size
            # Calculate the size of the cropped square
            size = min(width, height)
            # Calculate the coordinates of the cropped square
            left = (width - size) / 2
            top = (height - size) / 2
            right = (width + size) / 2
            bottom = (height + size) / 2

            # Crop the image to a square and resize it to 400x400 pixels
            img = img.crop((left, top, right, bottom)).resize((400, 400))
            # Save the cropped and resized image
            img.save(output_path)

            # Add the image path and time point to the data association dictionary
            data_association[output_path] = time_point
            # Add the time point to the labels list
            labels.append([time_point])

# Save the data association and labels to a .npz file
np.savez(os.path.join(output_dir, 'data_association_mandelbrot.npz'), filepaths=list(data_association.keys()), labels=labels)
# Release the video file
video.release()