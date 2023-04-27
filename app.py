import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(r'CKD.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('main.html')


@app.route('/Home')
def my_home():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['blood_urea', 'blood glucose random', 'anemia',
                     'coronary_artery_disease', 'pus_cell', 'red_blood_cell',
                     'diabetesmellitus', 'pedal_edema']
    df = pd.DataFrame(features_value, columns=features_name)
    print(df)
    predit = model.predict(df)
    print(predit)
    if predit == 0:
        return render_template("main.html", request="Oops! You have Chronic Kidney Disease")
    else:
        return render_template("main.html", request="Great! You Don't have Chronic Kidney Disease")


if __name__ == '__main__':
    app.run(debug=False)
