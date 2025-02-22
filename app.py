from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your machine learning model here
model = tf.keras.models.load_model('mnist_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['image']
    image = Image.open(uploaded_file)
    image = image.resize((28, 28))  # Resize image if needed
    image = np.asarray(image) / 255.0  # Normalize

# If your model expects a batch, you can use np.expand_dims
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Convert predicted_class to a Python int
    predicted_class = int(predicted_class)

    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
