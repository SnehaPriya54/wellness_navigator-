import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model on top of the pre-trained base model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes (Medicinal Leaf dataset and Medicinal plant dataset)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Define data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set up the data flow from directories
train_generator = train_datagen.flow_from_directory('train',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='sparse')

validation_generator = test_datagen.flow_from_directory('test',
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='sparse')

history = model.fit(train_generator,
                    epochs=10,
                    validation_data=validation_generator)

# Assuming 'model' is your trained model
model.save('medicinal_plants_model.h5')

