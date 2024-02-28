import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


df = pd.read_csv("./app/pages/rmseplt.csv")
chart_data = pd.DataFrame(df["Epochs"], df["rmse_values_train"])

st.line_chart(chart_data)
