import streamlit as st
import numpy as np
import pandas as pd



chart_data = pd.DataFrame([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], [7.944234761966651,
 7.344119762294723,
 7.1997220161386215,
 7.151222034236081,
 7.121901498210893,
 7.099689608510607,
 7.0823707166259675,
 7.06882998027756,
 7.058211810170119,
 7.0498692020033005])

st.line_chart(chart_data)
