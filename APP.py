import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle

model = pickle.load(open('egypt_house_price_prediction.pickle','rb'))
house_data = pd.read_csv('house2.csv')



st.write("""
<style>
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
st.title("توقع سعر العقارات ")
st.title(" في مصر")
city = st.selectbox('المحافظة', house_data['city'].unique())
location1 = st.selectbox('المنطقة', house_data[house_data['city']==city]['location1'].unique())
unit = st.selectbox('نوع الوحدة', house_data['unit'].unique())
bedrooms = st.selectbox('غرف النوم',range(1,16))
bathrooms = st.selectbox('حمام',range(1,11))
size = st.number_input('(م2)المساحة')


def predict_house_price(bathrooms,unit,location1,bedrooms,size):
    data = {'bedrooms': [bedrooms], 'size': [size], 'unit': [unit], 'location1': [location1], 'bathrooms': [bathrooms]}
    df = pd.DataFrame(data)
    df=df.iloc[:,:].values
    all_locations=list( house_data.location1.unique())
    all_units=list (house_data.unit.unique())
    encoded_location1 = [1 if location1 == r else 0 for r in all_locations]
    encoded_units = [1 if unit == i else 0 for i in all_units]
    features = [[bathrooms]+encoded_units + encoded_location1 + [bedrooms, size]]
    #features = [encoded_units + encoded_location1 + [bedrooms, size,bathrooms]]
   # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [2,3])], remainder='passthrough')
   # x = np.array(ct.fit_transform(df))  
    # One-hot encode the categorical features
    #df = pd.get_dummies(df, columns=['location1', 'unit'])
    
    # Make a prediction using the input features
    predicted_price =  model.predict(features)[0]
    
    return predicted_price
if st.button("Predict"):
    # Make predictions using your model
    prediction = predict_house_price(bathrooms,unit,location1,bedrooms,size)
    # Display the prediction
    st.write("<div style='text-align:center;font-size: 26px;'>(جم)السعر المتوقع</div>", unsafe_allow_html=True)
    st.write(f"<div style='text-align:center; color: red;font-size: 60px;'>{int(prediction)}</div>", unsafe_allow_html=True)
    