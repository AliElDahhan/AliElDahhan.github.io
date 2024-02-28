import streamlit as st
import pandas as pd
import numpy as np

st.line_chart(
   chart_data, x=df["Epochs"], y=df["Epochs"], color=["#FF0000", "#0000FF"]  # Optional
)
