import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")
pickle_in = open("randomforest.pkl", 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    This gives the results from the rendering of the HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)

    return render_template('index.html', predict_text='The type of seed belongs to: {}'.format(str(prediction)))


if __name__ == '__main__':
    app.run(debug=True)