# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:12:38 2022

@author: bh04
"""


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

#Visualizations
st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Caching to avoid reading the CSV file every time, unless the file name is changed
@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data

with header: 
    st.title("Welcome to my first streamlet project!")
    st.text('In this project I look into the transactions of taxis in NYC. ...')

with dataset:
    st.header('NY Taxi Dataset')
    st.text('I found this dataset on blablabla.com, ...')
    
    taxi_data = get_data('C:/Users/bh04/Desktop/Streamlit/Project 1/data/taxi_data.csv')
    #taxi_data = pd.read_csv('C:/Users/bh04/Desktop/Streamlit/Project 1/data/taxi_data.csv',low_memory=False) 
    # I added the low_memory=False because the data is big for loading
    #st.write(taxi_data.head())
    
    st.subheader('Pick-up Location ID distribution on the NYC dataset')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)
    

with features:
    st.header('The features I created')
    
    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic...')
    st.markdown('* **Second feature:** I created this feature because of this... I calculated it using this logic...')

with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')

    sel_col, disp_col = st.beta_columns(2)
    
    max_depth = sel_col.slider('Wha should be the max_depth of the model?', min_value=10, max_value=100, value =20, step=10)
    
    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No Limit'], index = 0)
    
    # display a list of available columns
    sel_col.text('Here is a list of features in my data')
    sel_col.write(taxi_data.columns)
    
    input_feature = sel_col.text_input('Wich feature should be used as the input features?','PULocationID')
    
    #ML module
    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    x = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]
    
    regr.fit(x,y.values.ravel()) #I added the .values.ravel() after getting an error A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel()
    prediction = regr.predict(y)
    
    # Display ML Results
    disp_col.subheader ('Meam absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))
    
    disp_col.subheader ('Meam squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))
    
    disp_col.subheader ('R squared score of the model is:')
    disp_col.write(r2_score(y, prediction))


