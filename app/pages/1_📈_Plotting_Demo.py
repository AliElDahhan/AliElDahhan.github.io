import streamlit as st
import numpy as np
import pandas as pd

chart_data = pd.DataFrame(
     [0.1, 0.5, 1.0],
     [0.1, 0.5, 0.9])

st.line_chart(chart_data)
