from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained CNN model (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Load the trained SVM model
svm = joblib.load(r'D:\Major project\svm_model.pkl')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Rescale the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    temp_file_path = os.path.join('uploads', file.filename)
    file.save(temp_file_path)

    # Call the preprocess_image function
    img_array = preprocess_image(temp_file_path)
    features = model.predict(img_array)
    features_flat = features.reshape((features.shape[0], -1))
    prediction = svm.predict(features_flat)

    os.remove(temp_file_path)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
