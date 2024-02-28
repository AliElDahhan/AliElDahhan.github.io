import streamlit as st
import numpy as np
import pandas as pd


Data = "https://github.com/AliElDahhan/AliElDahhan.github.io/blob/master/app/pages/rmseplt.csv"
data = pd.read_csv(DATA_URL)

chart_data = pd.DataFrame(
   {
      data["Epochs"], data["rmse_values_train"] 
 
)

st.line_chart(chart_data)

