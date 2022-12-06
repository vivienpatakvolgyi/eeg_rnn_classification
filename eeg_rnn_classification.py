import streamlit as st
import pandas as pd
import numpy as np

st.title('RNN classification with EEG data')
st.write("The original dataset is available from [here](https://www.kaggle.com/datasets/fabriciotorquato/eeg-data-from-hands-movement)")

train_test = st.radio(
    "I used two different approaches, each has different outcome, so please select which one you want to use for prediction:",
    ('a) We train the model on the data of three of the four users and predict on the remaining 1', 'b) The data used for prediction or training are randomly selected from the time series data'))

results = st.button('Show results', key='res')

if results: 

    if 'a)' in train_test:
        st.write('You chosed the A option. [Open original notebook](https://colab.research.google.com/drive/1wMTgQV1En_W1eQONhEyipMb2i9EmIbZ4?usp=sharing)')
        st.write('In this case, you can upload the fourth user\'s data in csv. You can download and see the original data I used for this project. After downloading the original .csv file, please drag and drop it here to start the prediction and see the results.')
    elif 'b)' in train_test:
        st.write('You chosed the B option. [Open original notebook](https://colab.research.google.com/drive/1JWahgKnkjCOrddkxIRy8vpQqH4QXnSkY?usp=sharing)')
        st.write('Sadly, in this case we don\'t have the data in csv, because it was from separation during the data processing. However, I\'m gonna show you the processed data structure right before we use it for prediction. In option a), after data processing, we have the same exact structure but with different data. This data is only the 20\% of the full dataset and it was selected randomly from all users time series data as sequences. In option a), we used 25% of the full dataset for prediction and it\'s from an entirely different user than the ones we used for training.')
        
        st.write('As you can see, the rate of correctly predicted values ​​is higher in this case, while in the first case we get a value of around 30%, which means that the prediction efficiency is the same as random guessing (considering that we have three categories).')

uploaded_file = st.file_uploader("Upload CSV", type=".csv")
st.write(uploaded_file)