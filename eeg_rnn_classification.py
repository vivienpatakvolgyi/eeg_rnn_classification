import streamlit as st
import pandas as pd
import numpy as np

st.title('RNN classification with EEG DATA')
uploaded_file = st.file_uploader("Upload CSV", type=".csv")
st.write(uploaded_file)