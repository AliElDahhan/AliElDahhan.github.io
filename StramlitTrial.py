#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample housing data (replace with your actual data)
housing = np.random.rand(100, 5)
housing_labels = np.random.rand(100)

# Define the app layout
def main():
    st.title('ElasticNet Model Analysis')
    page = st.sidebar.selectbox("Choose a page", ["Code", "Graphs"])
    
    if page == "Code":
        show_code()
    elif page == "Graphs":
        show_graphs()

# Function to display the Python code
def show_code():
    st.header("Python Code for the Project")
    st.code("""
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

alphas = [0.1, 0.5, 1.0]
l1_ratios = [0.1, 0.5, 0.9]

rmse_values = []

for alpha in alphas:
    for l1_ratio in l1_ratios:
        elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        pipeline = make_pipeline(elastic_net)
        pipeline.fit(housing, housing_labels)
        housing_predictions = pipeline.predict(housing)
        rmse = mean_squared_error(housing_labels, housing_predictions, squared=False) / 10000
        rmse_values.append((alpha, l1_ratio, rmse))

rmse_values = np.array(rmse_values)

plt.figure()
for l1_ratio in l1_ratios:
    rmse_vals = [entry[2] for entry in rmse_values if entry[1] == l1_ratio]
    plt.plot(alphas, rmse_vals, label=f'l1_ratio={l1_ratio}')
plt.xlabel('Alpha')
plt.ylabel('RMSE (scaled by 1/10000)')
plt.title('Alpha vs. RMSE')
plt.legend()
plt.show()

plt.figure()
for alpha in alphas:
    rmse_vals = [entry[2] for entry in rmse_values if entry[0] == alpha]
    plt.plot(l1_ratios, rmse_vals, label=f'Alpha={alpha}')
plt.xlabel('l1_ratio')
plt.ylabel('RMSE (scaled by 1/10000)')
plt.title('l1_ratio vs. RMSE')
plt.legend()
plt.show()
        """)

# Function to display the graphs
def show_graphs():
    st.header("Graphs")
    plt.figure()
    for l1_ratio in l1_ratios:
        rmse_vals = [entry[2] for entry in rmse_values if entry[1] == l1_ratio]
        plt.plot(alphas, rmse_vals, label=f'l1_ratio={l1_ratio}')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE (scaled by 1/10000)')
    plt.title('Alpha vs. RMSE')
    plt.legend()
    st.pyplot(plt)

    plt.figure()
    for alpha in alphas:
        rmse_vals = [entry[2] for entry in rmse_values if entry[0] == alpha]
        plt.plot(l1_ratios, rmse_vals, label=f'Alpha={alpha}')
    plt.xlabel('l1_ratio')
    plt.ylabel('RMSE (scaled by 1/10000)')
    plt.title('l1_ratio vs. RMSE')
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    main()


# In[ ]:




