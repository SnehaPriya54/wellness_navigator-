import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_main_sub_types_names(dataset_path):
    main_types = []
    subtypes = []

    for main_type_folder in os.listdir(dataset_path):
        main_types.append(main_type_folder)
        main_type_path = os.path.join(dataset_path, main_type_folder)

        if os.path.isdir(main_type_path):
            for subtype_folder in os.listdir(main_type_path):
                subtypes.append(subtype_folder)

    return main_types, subtypes

# Specify the path to your dataset folder
dataset_path = 'dataset'


# Define the paths to your dataset folders
train_data_dir = 'train'
validation_data_dir = 'test'

# Get the number of main types and subtypes
main_types, subtypes = get_main_sub_types_names(train_data_dir)
num_main_types = len(main_types)
num_subtypes = len(subtypes)

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add global average pooling layer and dropout for regularization
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)

# Add main type prediction layer
main_type_output = Dense(num_main_types, activation='softmax', name='main_type')(x)

# Add subtype prediction layer
subtype_output = Dense(num_subtypes, activation='softmax', name='subtype')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=[main_type_output, subtype_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'main_type': 'sparse_categorical_crossentropy',
                    'subtype': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='sparse')

# Train the model
history = model.fit(train_generator,
                    epochs=1,
                    validation_data=validation_generator)

# Save the model
model.save('medicinal_plants_model.h5')
