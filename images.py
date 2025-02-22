from PIL import Image
import os

import tensorflow as tf

# Load the MNIST dataset
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Create a directory to save the images
os.makedirs('mnist_images', exist_ok=True)

# Loop through the test images and save them
for i, image_data in enumerate(x_test):
    # Convert to Image object
    image = Image.fromarray(image_data)

    # Save the image with a filename like 'mnist_image_0.png'
    image.save(f'mnist_images/mnist_image_{i}.png')
