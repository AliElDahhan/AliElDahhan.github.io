import streamlit as st

def show_page1():
    st.title("Python Code for the Project")

    code = """
    # Your Python code for the project goes here
    """
    st.code(code, language='python')
