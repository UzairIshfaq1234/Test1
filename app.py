from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Brest_CNN.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['file']
        
        # Read the image
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (50, 50))
        img = np.reshape(img, [1, 50, 50, 3])

        # Make a prediction
        prediction = model.predict(img)
        result = int(np.argmax(prediction))  # Convert to standard Python int

        # Return the result
        return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
