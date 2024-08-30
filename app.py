from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
filename = 'knn.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the web form
    age = request.form.get('age')
    anaemia = request.form.get('anaemia')
    creatinine_phosphokinase = request.form.get('creatinine_phosphokinase')
    diabetes = request.form.get('diabetes')
    ejection_fraction = request.form.get('ejection_fraction')
    high_blood_pressure = request.form.get('high_blood_pressure')
    platelets = request.form.get('platelets')
    serum_creatinine = request.form.get('serum_creatinine')
    serum_sodium = request.form.get('serum_sodium')
    sex = request.form.get('sex')
    smoking = request.form.get('smoking')


    features = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking]

    # Convert the list to a 2D array
    features = [list(map(int, features))]

    # Predict the class using the model
    prediction = model.predict(features)[0]


    # Render a new web page with the prediction
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)