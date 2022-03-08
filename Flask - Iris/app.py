# Bibliotecas
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Inicializa the flask App
model = pickle.load(open('model.pkl', 'rb')) # carrega o modelo

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # recupera os valores do formulário
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # faz a predição


    return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction)) # renderiza o resultado da predição

if __name__ == "__main__":
    app.run(debug=True)