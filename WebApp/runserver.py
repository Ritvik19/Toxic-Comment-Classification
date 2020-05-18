from flask import Flask, request, jsonify, render_template
from application import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict',methods=['POST'])
def predict():

    inp = str(list(request.form.values())[0])
    features = generateFeatures(inp)
    prediction = getPredictions(features, model0)
    
    identity_hate = getPredictions(features, models[0])
    insult = getPredictions(features, models[1])
    obscene = getPredictions(features, models[2])
    severe_toxic = getPredictions(features, models[3])
    threat = getPredictions(features, models[4])
    toxic = getPredictions(features, models[5])
    
    vader = getVader(inp)
    
    prediction_text = {
        'Positive' : list(map(lambda x: round(x*100, 2), prediction)),
        'IdentityHate' :  list(map(lambda x: round(x*100, 2), identity_hate)),
        'Insult':  list(map(lambda x: round(x*100, 2), insult)),
        'Obscene':  list(map(lambda x: round(x*100, 2), obscene)),
        'SevereToxic':  list(map(lambda x: round(x*100, 2), severe_toxic)),
        'Threat':  list(map(lambda x: round(x*100, 2), threat)),
        'Toxic':  list(map(lambda x: round(x*100, 2), toxic)),
        'Vader': list(map(lambda x: round(x*100, 2), vader)),
    }

    return render_template('index.html', inp=inp, prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)