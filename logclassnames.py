import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = tf.keras.models.load_model('image_classifier_model.h5')

# Define paths
train_data_dir = 'train'
validation_data_dir = 'test'

# Image dimensions and other hyperparameters
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
batch_size = 32
epochs = 5

# Load class names
class_names = sorted(os.listdir(train_data_dir))

# Create a dictionary to map class indices to names
class_index_to_name = {i: class_name for i, class_name in enumerate(class_names)}

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build and compile the model (you can reuse your existing model loading and compiling code here)
model.compile(
    optimizer='adam',  # Use the same optimizer as before
    loss='categorical_crossentropy',  # Use the same loss function as before
    metrics=['accuracy']
)

# Assuming model is already loaded and compiled

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    print('-' * 10)
    
    for i, (inputs, labels) in enumerate(train_generator):
        # Training step
        loss, acc = model.train_on_batch(inputs, labels)
        
        # Log class index and name
        class_index = np.argmax(labels[0])
        class_name = class_index_to_name[class_index]
        
        print(f"Batch {i+1}: Loss: {loss:.4f} - Accuracy: {acc:.4f} - Class: {class_index} ({class_name})")
        
        if i+1 == len(train_generator):
            break
