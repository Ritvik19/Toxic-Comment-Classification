from flask import Flask, request, jsonify, render_template
import pickle

import numpy as np
from sklearn.base import BaseEstimator

import nltk, re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

class TextCleaner(BaseEstimator):
    
    def __init__(self, rsw=True, stm=False, lem=False):
        
        self.rsw = rsw
        self.stm = stm
        self.lem = lem
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.ps = PorterStemmer()
        self.wnl = WordNetLemmatizer()
        
    def fit(self, x, y=None):
        return self
    
    def spell_correct(self, text):
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"what's", "what is ", text) 
        text = re.sub(r"'s", " ", text)
        text = re.sub(r"'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"'re", " are ", text)
        text = re.sub(r"'d", " would ", text)
        text = re.sub(r"'ll", " will ", text)
        text = re.sub(r"s", "0", text)    
        return text

    def remove_url(self, text):
        URL_REGEX = re.compile(r'''((http[s]?://)[^ <>'"{}|\^`[\]]*)''')
        return URL_REGEX.sub(r' ', text)

    def remove_handles(self, text):
        HANDLES_REGEX = re.compile(r'@\S+')
        return HANDLES_REGEX.sub(r' ', text)

    def remove_incomplete_last_word(self, text):
        INCOMPLETE_LAST_WORD_REGEX = re.compile(r'\S+…')
        return INCOMPLETE_LAST_WORD_REGEX.sub(r' ', text )

    def remove_punc(self, text):
        return re.sub(r"\W", ' ', text)

    def remove_num(self, text):
        return re.sub(r"\d", ' ', text)

    def remove_extra_spaces(self, text):
        return re.sub(r"\s+", ' ', text).strip()

    def remove_shortwords(self, text): 
        return ' '.join(word for word in text.split() if len(word) > 2)

    def lower_case(self, text):
        return  text.lower()

    def remove_stopwords(self, text):
        return ' '.join(word for word in text.split() if word not in self.stop_words)

    def ps_stem(self, text):
        return ' '.join(self.ps.stem(word) for word in text.split())

    def wnl_lemmatize(self, text):
        return ' '.join(self.wnl.lemmatize(word) for word in text.split())

    def clean(self, x, rsw, stm, lem):
        x = str(x)
        x = self.remove_url(str(x))
        x = self.lower_case(str(x))
        x = self.spell_correct(str(x))
        x = self.remove_punc(str(x))
        x = self.remove_num(str(x))
        x = self.remove_extra_spaces(str(x))
        x = self.remove_shortwords(str(x))

        if rsw:
            x = self.remove_stopwords(str(x))
        if stm:
            x = self.ps_stem(str(x))
        if lem:
            x = self.wnl_lemmatize(str(x))
        return x
    
    def transform(self, x):
        x = map(lambda text: self.clean(text, self.rsw, self.stm, self.lem)  , x)
        x = np.array(list(x))
        return x


vectorizer, model0, *models = pickle.load(open('OverSampling.pkl', 'rb'))
sia_obj = SentimentIntensityAnalyzer() 

def generateFeatures(inp):
    features = vectorizer.transform([str(inp)])
    return features

def getPredictions(feats, model):
    return model.predict_proba(feats)[0]

def getVader(inp):
    sentiment_dict = sia_obj.polarity_scores(inp) 
    return [sentiment_dict['neg'], sentiment_dict['neu'] , sentiment_dict['pos']]
    
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