import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler
import joblib


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def result():
    gender = int(request.form['gender'])
    ssc_p = float(request.form['ssc_p'])
    ssc_b = str(request.form['ssc_b'])
    hsc_p = float(request.form['hsc_p'])
    hsc_b = str(request.form['hsc_b'])
    hsc_s = str(request.form['hsc_s'])
    degree_p = float(request.form['degree_p'])
    degree_t = str(request.form['degree_t'])
    workex = str(request.form['workex'])
    etest_p = float(request.form['etest_p'])
    specialisation = str(request.form['specialisation'])
    mba_p = float(request.form['mba_p'])

    x = np.array(
        [gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p])

    columns = ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p',
               'specialisation', 'mba_p']

    data = pd.DataFrame(x.reshape(1, 12), columns=columns)
    # Encoding features
    file_name = 'models/transformer'
    model = joblib.load(file_name)
    # Loading Model
    model_file = 'models/best_model'
    estimator = joblib.load(model_file)
    best_estimator = estimator.best_estimator_

    x_new = pd.DataFrame(model.transform(data))
    x_scaled = StandardScaler().fit_transform(x_new)
    prediction = estimator.predict(x_scaled)
    if prediction == 1:
        prediction = 'Placed'
    else:
        prediction = 'not placed'
    return jsonify({'Prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True, port=9457)
