from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("crop_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    rainfall = float(request.form['rainfall'])
    temperature = float(request.form['temperature'])
    soil = request.form['soil']
    crop = request.form['crop']
    fertilizer = float(request.form['fertilizer'])
    area = float(request.form['area'])

    soil_encoded = soil_encoder.transform([soil])[0]
    crop_encoded = crop_encoder.transform([crop])[0]

    input_data = np.array([[rainfall, temperature, soil_encoded, crop_encoded, fertilizer, area]])

    prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True)
