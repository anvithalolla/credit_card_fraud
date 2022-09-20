import pandas as pd
import numpy as np
import pickle

from flask import Flask, request, render_template

app = Flask(__name__, template_folder='C:/Users/anvil/OneDrive/Documents/flaskapp/project/CCFD/templates')
load_model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    test_input = [float(x) for x in request.form.values()]
    final_input = pd.DataFrame(np.array(test_input).reshape(-1,len(test_input)), columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'scaled_amount', 'scaled_time'])
    prediction = load_model.predict(final_input)

    output = ''
    if(prediction[0] == 0):
        output = 'Fraudulent'
    else:
        output = 'Normal'
        
    return render_template('index1.html', prediction_text = 'The transaction is ' + output)
 

if (__name__ == "__main__"):
    app.run(debug = True)



