from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)

model=pickle.load(open('lr_model.pkl','rb'))
car=pd.read_csv('Cleaned_Car.csv')
car.drop(columns='Unnamed: 0',inplace=True)
gold_cars=[]
for i in range(len(car)):
    if car.loc[i,'Label']=='GOLD':
        gold_cars.append(car.loc[i,'Name'])

@app.route('/',methods=['GET','POST'])
@cross_origin()
def index():
    companies = sorted(car['Company'].unique())
    car_names = sorted(car['Name'].unique())
    year = sorted(car['Year'].unique(), reverse=True)
    fuel_type = car['Fuel_type'].unique()
    owner=car['Owner'].unique()
    return render_template('index.html', companies=companies, car_models=car_names, years=year, fuel_types=fuel_type, owners=owner)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    company=request.form.get('company')
    car_name = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    owner=request.form.get('owner')
    km=int(request.form.get('kilo_driven'))
    if car_name in gold_cars:
        label=0
    else:
        label=1
    test = [[car_name, label, km, fuel_type, owner, year, company]]
    prediction=model.predict(pd.DataFrame(data=test,columns=['Name', 'Label', 'Kms_driven', 'Fuel_type', 'Owner', 'Year','Company']))
    print(prediction)
    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)