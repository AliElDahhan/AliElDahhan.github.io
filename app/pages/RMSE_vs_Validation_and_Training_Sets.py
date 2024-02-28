import streamlit as st
import pandas as pd
import numpy as np

chart_data = pd.DataFrame("rmseplt.csv", columns=["a", "b", "c"])

st.line_chart(chart_data)
