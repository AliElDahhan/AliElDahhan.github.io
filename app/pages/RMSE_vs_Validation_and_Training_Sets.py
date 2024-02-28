import streamlit as st
import pandas as pd

# Hard-coded vectors
epochs = list(range(1, 6))  # Example epochs [1, 2, 3, 4, 5]
rmse_train = [7.94, 7.34, 7.19, 7.15, 7.12]  # Example RMSE values for training
rmse_valid = [7.90, 7.33, 7.19, 7.16, 7.14]  # Example RMSE values for validation

# Create a DataFrame
data = {
    'Epochs': epochs,
    'RMSE Train': rmse_train,
    'RMSE Valid': rmse_valid
}
df = pd.DataFrame(data)

# Plot the RMSE values
st.line_chart(df.set_index('Epochs'))
