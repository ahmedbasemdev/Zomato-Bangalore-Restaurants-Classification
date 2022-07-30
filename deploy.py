from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

fullPipeline = joblib.load('pipline.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():
    features = [1,1,775,800,'Buffet','Banashankari',30,'Casual Dining']
    dataColumns = ['online_order','book_table','votes','cost','type','city','cuisinesCount','rest_type']
    data = pd.DataFrame(data=features)
    data = data.T
    data.columns = dataColumns


    prediction = fullPipeline.predict(data)

    #final_features = [np.array(features)]

    
    return render_template('predict.html',predcitionValue=prediction)

if __name__ == '__main__':
    app.run(debug=True)
