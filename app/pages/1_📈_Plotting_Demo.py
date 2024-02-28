import streamlit as st
import numpy as np
import pandas as pd



df = pd.read_csv("rmseplt.csv")
chart_data = pd.DataFrame(df["Epochs"], df["rmse_values_train"])

st.line_chart(chart_data)
