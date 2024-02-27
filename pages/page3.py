#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show_page3():
    st.title("Model Fit to the Data")

    # Dummy data for demonstration
    X = np.linspace(-5, 5, 100)
    y = 2 * X + 1
    y_pred = 2 * X + 1 + np.random.normal(0, 1, 100)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label='Actual Data')
    plt.plot(X, y_pred, color='red', label='Model Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

