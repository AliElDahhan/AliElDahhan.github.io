#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt

def show_page2():
    st.title("RMSE vs Epoch")

    # Dummy data for demonstration
    epochs = range(1, 11)
    train_rmse = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    val_rmse = [1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_rmse, label='Training RMSE')
    plt.plot(epochs, val_rmse, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

