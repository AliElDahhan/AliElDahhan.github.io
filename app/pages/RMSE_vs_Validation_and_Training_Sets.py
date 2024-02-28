import streamlit as st
import pandas as pd
import numpy as np



chart_data = pd.DataFrame([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], columns=["Epochs", "rmse_values_train", "rmse_values_vaild"])

st.line_chart(
   chart_data, x="Epochs", y=["rmse_values_train", "rmse_values_vaild"], color=["#FF0000", "#0000FF"]  # Optional
)



