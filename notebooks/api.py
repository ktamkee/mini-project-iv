from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy
import pickle

app = Flask(__name__)
api = Api(app)

class DataframeFunctionTransformer:
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

def create_total_income_feature(input_df):
    input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    return input_df

def to_dataframe(array):
    columns= ['Gender','Dependents','Married','Self_Employed', 'LoanAmount',
               'Loan_Amount_Term','Credit_History','Education','ApplicantIncome',
               'CoapplicantIncome','Property_Area', 'TotalIncome']
    
    return pd.DataFrame(array, columns = columns)
    
    return pd.DataFrame(array, columns = columns)

def log_object(input_df):
    input_df['LoanAmount'] = np.log(input_df['LoanAmount'])
    input_df['TotalIncome'] = np.log(input_df['TotalIncome'])
    return input_df

model = pickle.load( open( "model.p", "rb" ) )

class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        res = model.predict_proba(df)
        return res.tolist() 

api.add_resource(Scoring, '/scoring')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)