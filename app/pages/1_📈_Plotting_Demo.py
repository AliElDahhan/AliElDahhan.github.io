import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv("./app/pages/rmseplt.csv")
DATE_COLUMN1 = 'Epochs'
DATE_COLUMN2 = 'rmse_values_train	'

fig = px.line(df, x="Epochs", y="rmse_values_train", title=inform)

st.plotly_chart(fig, use_container_width=True)
