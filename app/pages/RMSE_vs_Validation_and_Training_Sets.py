import streamlit as st
import pandas as pd
import numpy as np

chart_data = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=["col1", "col2", "col3"])

st.line_chart(
   chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional
)
