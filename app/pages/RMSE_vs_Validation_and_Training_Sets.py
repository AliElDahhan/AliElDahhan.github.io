# page2.py
import streamlit as st
import pandas as pd

def app():
    st.title("Page 2")
    # Your code for Page 2 (e.g., plotting RMSE values)
    df = pd.read_csv('rmseplt.csv')
    st.line_chart(df.set_index('Epochs')[['rmse_values_train', 'rmse_values_valid']])
