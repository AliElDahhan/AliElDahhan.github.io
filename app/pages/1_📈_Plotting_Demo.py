import streamlit as st
import numpy as np
import pandas as pd

filename = "rmseplt.csv"
df = pd.read_csv(filename)

chart_data = pd.DataFrame(
     df.iloc[:, 0], df.iloc[:, 1])

st.line_chart(chart_data)
