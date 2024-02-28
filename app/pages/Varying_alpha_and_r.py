import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('rmseplt.csv')

st.line_chart(df.set_index('Epochs')[['rmse_values_train', 'rmse_values_valid']])
