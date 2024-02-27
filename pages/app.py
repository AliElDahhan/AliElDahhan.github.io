#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from page1 import show_page1
from page2 import show_page2
from page3 import show_page3

PAGES = {
    "Python Code": show_page1,
    "RMSE vs Epoch": show_page2,
    "Model Fit": show_page3
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()

