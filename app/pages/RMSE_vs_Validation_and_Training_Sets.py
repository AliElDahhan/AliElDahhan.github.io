# page2.py
import streamlit as st
import pandas as pd

def app():
    st.title("Page 2")
    # Load and display the data
    df = pd.read_csv('rmseplt (1).csv')
    st.write(df)  # Check if the data is displayed correctly

    # Plot the RMSE values
    st.line_chart(df.set_index('Epochs')[['rmse_values_train', 'rmse_values_valid']])
