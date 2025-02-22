from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
leaf_name = { "0":"(Aloevera)",
"1":"(Amla)",
"10":"(Bhrami)",
"11":"(Bringaraja)",
"12":"(Caricature)",
"13":"(Castor)",
"14":"(Catharanthus)",
"15":"(Chakte)",
"16":"(Chilly)",
"17":"(Citron lime (herelikai))",
"18":"(Coffee)",
"19":"(Common rue(naagdalli))",
"2":"(Amruthaballi)",
"20":"(Coriender)",
"21":"(Curry)",
"22":"(Doddpathre)",
"23":"(Drumstick)",
"24":"(Ekka)",
"25":"(Eucalyptus)",
"26":"(Ganigale)",
"27":"(Ganike)",
"28":"(Gasagase)",
"29":"(Ginger)",
"3":"(Arali)",
"30":"(Globe Amarnath)",
"31":"(Guava)",
"32":"(Henna)",
"33":"(Hibiscus)",
"34":"(Honge)",
"35":"(Insulin)",
"36":"(Jackfruit)",
"37":"(Jasmine)",
"38":"(Kambajala)",
"39":"(Kasambruga)",
"4":"(Astma_weed)",
"40":"(Kohlrabi)",
"41":"(Lantana)",
"42":"(Lemon)",
"44":"(Malabar_Nut)",
"45":"(Malabar_Spinach)",
"46":"(Mango)",
"47":"(Marigold)",
"48":"(Mint)",
"49":"(Neem)",
"5":"(Badipala)",
"50":"(Nelavembu)",
"51":"(Nerale)",
"52":"(Nooni)",
"53":"(Onion)",
"54":"(Padri)",
"55":"(Palak(Spinach))",
"56":"(Papaya)",
"57":"(Parijatha)",
"58":"(Pea)",
"59":"(Pepper)",
"6":"(Balloon_Vine)",
"60":"(Pomoegranate)",
"61":"(Pumpkin)",
"62":"(Raddish)",
"63":"(Rose)",
"64":"(Sampige)",
"65":"(Sapota)",
"66":"(Seethaashoka)",
"67":"(Seethapala)",
"68":"(Spinach1)",
"69":"(Tamarind)",
"7":"(Bamboo)",
"70":"(Taro)",
"71":"(Tecoma)",
"72":"(Thumbe)",
"73":"(Tomato)",
"74":"(Tulsi)",
"75":"(Turmeric)",
"76":"(ashoka)",
"77":"(camphor)",
"78":"(kamakasturi)",
"79":"(kepala)",
"8":"(Beans)",
"9":"(Betel)"
}
# Load your machine learning model here
model = tf.keras.models.load_model('image_classifier_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['image']
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  # Resize image if needed
    image = np.asarray(image) / 255.0  # Normalize

    # If your model expects a batch, you can use np.expand_dims
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Convert predicted_class to a Python int
    predicted_class = int(predicted_class)
    print(predicted_class)

    return jsonify({'prediction': leaf_name[str(predicted_class)]})


if __name__ == '__main__':
    app.run(debug=True)
