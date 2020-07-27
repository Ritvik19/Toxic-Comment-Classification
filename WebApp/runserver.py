from flask import Flask, request, jsonify, render_template
from application import *
import json
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', results='')

@app.route('/predict',methods=['POST'])
def predict():

    inp = str(list(request.form.values())[0])
    features1 = generateFeatures(inp, vectorizer1)
    features2 = generateFeatures(inp, vectorizer2)
    
    model1_pred = list(getPredictions(features1, model_1))
    model2_pred = list(getPredictions(features2, model_2))
            
    vader = getVader(inp)
    
    results ={
        'overview' : {
            'positive': list(map(
                lambda x: round(x*100, 2), 
                [model1_pred[0], model2_pred[0], (model1_pred[0]+ model2_pred[0])/2]
            )),
            'negative': list(map(
                lambda x: round((1-x)*100, 2), 
                [model1_pred[0], model2_pred[0], (model1_pred[0]+ model2_pred[0])/2]
            ))
        },
        'vader': vader,
        'model1': list(map(lambda x: round(x*100, 2), model1_pred[1:])),
        'model2': list(map(lambda x: round(x*100, 2), model2_pred[1:])),

    }
    
    return render_template('index.html', inp=inp, results=results, debug=0)

if __name__ == "__main__":
    app.run(debug=True)