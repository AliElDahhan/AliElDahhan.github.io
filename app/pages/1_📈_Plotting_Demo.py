import streamlit as st
import numpy as np
import pandas as pd

DATE_COLUMN1 = 'Epochs'
DATE_COLUMN2 = 'rmse_values_train	'
DATA_URL = (https://github.com/AliElDahhan/AliElDahhan.github.io/blob/master/app/pages/rmseplt.csv)

chart_data = pd.DataFrame(
     DATE_COLUMN1,DATE_COLUMN2)

st.line_chart(chart_data)
