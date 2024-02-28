import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv("./app/pages/rmseplt.csv")
DATE_COLUMN1 = 'Epochs'
DATE_COLUMN2 = 'rmse_values_train	'


chart_data = pd.DataFrame(
     df.iloc[:, 1],df.iloc[:, 2])

st.line_chart(chart_data)
