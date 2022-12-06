import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Reshape
from numpy import mean
from numpy import std
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import tensorflow_addons as tfa
import pickle

def append_time_series(df):
  X = []
  Y = []
  size= 1
  #df_y= df_y.drop('UserId', axis =1)
  for i in range(0,len(df)-size, 1):
    X.append(np.array(df[i:i+size].drop(df.columns[112:], axis =1)).reshape(112))
    label=df.values[i+size][-1].astype(float)
    Y.append(float(label))
    
  X= np.array(X)
  Y= np.array(Y)
  
  return X, Y

def MSE_R(true, prediction):
    st.write(f"MSE: %.3f" % mean_squared_error(true, prediction))
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(np.array(true), np.array(prediction))
    result = metric.result()
    st.write(f"R\u00B2: %.3f" % result.numpy())

def show_results(true, prediction):
    results = pd.DataFrame()
    results['Prediction'] = prediction
    results['True value'] = true
    st.write(results)

def predict(X_predict, y_predict, file = ''):
    fname = file
    model = load_model(fname)

    predicted_vals = []

    for i in range(len(X_predict)):
        pred_test = model.predict(X_predict[i])[-1]
        predicted_vals.append(pred_test)

    true = []
    prediction = []

    correct = 0
    for i in range(len(predicted_vals)):
        prediction.append(list(predicted_vals[i]).index(max(predicted_vals[i])))
        true.append(int(y_predict[i][-1]))
        if int(list(predicted_vals[i]).index(max(predicted_vals[i]))) == int(y_predict[i][-1]):
            correct += 1

    st.write("Correct predictions: ", "{:.0%}".format(correct/len(true)))

    
    show_results(true, prediction)
    MSE_R(true, prediction)
st.title('RNN classification with EEG data')
st.write("The original dataset is available from [here](https://www.kaggle.com/datasets/fabriciotorquato/eeg-data-from-hands-movement)")

train_test = st.radio(
    "I used two different approaches, each has different outcome, so please select which one you want to use for prediction:",
    ('a) We train the model on the data of three of the four users and predict on the remaining 1', 'b) The data used for prediction or training are randomly selected from the time series data'))


if 'a)' in train_test:
    st.write('You chosed the A option. [Open original notebook](https://colab.research.google.com/drive/1wMTgQV1En_W1eQONhEyipMb2i9EmIbZ4?usp=sharing)')
    st.write('In this case, you can upload the fourth user\'s data in csv. You can download and see the original data I used for this project. After downloading the original .csv file, please drag and drop it here to start the prediction and see the results.')
    with open('user_d.csv', 'rb') as f:
        st.download_button('Download csv file', f, file_name='user_d.csv')
    uploaded_file = st.file_uploader("Upload csv file", type=".csv")
    if uploaded_file:
        item = pd.DataFrame(pd.read_csv(uploaded_file))
        scaler = StandardScaler()
        cols = item['Class']
        df = pd.DataFrame(scaler.fit_transform(item.drop(['Class'], axis = 1)),columns=item.columns.drop(['Class']))
        df['Class'] =cols
        
        X_val, y_val = append_time_series(df)
        generator_2 = TimeseriesGenerator(X_val, y_val, length=15, batch_size=32, shuffle = True)
        

        X_predict = []
        y_predict = []
        for i in range(len(generator_2)):
            X_predict.append(generator_2[i][0])
            y_predict.append(generator_2[i][1])

        model = load_model('RNN_3of4.h5')

        predicted_vals = []


        for i in range(len(X_predict)):
            pred_test = model.predict(X_predict[i])[-1]
            predicted_vals.append(pred_test)

        true = []
        prediction = []

        correct = 0
        for i in range(len(predicted_vals)):
            prediction.append(list(predicted_vals[i]).index(max(predicted_vals[i])))
            true.append(int(y_predict[i][-1]))
            if int(list(predicted_vals[i]).index(max(predicted_vals[i]))) == int(y_predict[i][-1]):
                correct += 1

        st.write("Correct predictions: ", "{:.0%}".format(correct/len(true)))

        
        show_results(true, prediction)
        MSE_R(true, prediction)
  
            
        

elif 'b)' in train_test:
    st.write('You chosed the B option. [Open original notebook](https://colab.research.google.com/drive/1JWahgKnkjCOrddkxIRy8vpQqH4QXnSkY?usp=sharing)')
    st.write('Sadly, in this case, we don\'t have the data in csv, because it was from separation during the data processing. However, I\'m going to show you the processed data structure right before we use it for prediction. In option a), after data processing, we\'ll have the same structure but different values. This data is only 20% of the full dataset and it was selected randomly from all users time series data as sequences. In option a), we used 25% of the full dataset for prediction and it\'s from an entirely different user than the ones we used for training.')
    X_predict = pickle.load( open( "validation_data_X", "rb" ) )
    y_predict = pickle.load( open( "validation_data_y", "rb" ) )

    predict(X_predict, y_predict, 'RNN_random.h5')


    st.write('As you can see, the rate of correctly predicted values ​​is higher in this case, while in the first case we get a value of around 30%, which means that the prediction efficiency is the same as random guessing (considering that we have three categories).')

