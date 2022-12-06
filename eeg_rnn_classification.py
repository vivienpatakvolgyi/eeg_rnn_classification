import streamlit as st
import pandas as pd
import numpy as np

st.title('RNN classification with EEG data')

train_test = st.radio(
    "I used two different approaches, each has different outcome, so please select which one you want to use for prediction:",
    ('a) We train the model on the data of three of the four users and predict on the remaining 1', 'b) The data used for prediction are randomly selected from the time series data'))

results = st.button('Show results', key='res')

if results: 

    if 'a)' in train_test:
        st.write('You chosed the A option')
    elif 'b)' in train_test:
        st.write('You chosed the B option')
        

uploaded_file = st.file_uploader("Upload CSV", type=".csv")
st.write(uploaded_file)