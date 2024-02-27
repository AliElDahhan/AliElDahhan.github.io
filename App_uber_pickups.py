# https://streamlit.io
# pip install streamlit
# https://saiharishcherukuri.medium.com/mastering-streamlit-essential-commands-for-interactive-apps-8ad570115f18
# https://drlee.io/a-comprehensive-guide-to-streamlit-cloud-building-interactive-and-beautiful-data-apps-af747bbac3e0
# streamlit run applicationname

import streamlit as st
import pandas as pd
import numpy as np

#Each App should have a title
st.title('Uber pickups in NYC')

#Load some data
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data   #avoids loading data each time you run the code
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()      #simply changes data to lowercase
    data.rename(lowercase, axis='columns', inplace=True)     #Rename columns or index labels
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])  #converts date column into datatime
    return data
    

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

data_load_state.text("Done! (using st.cache_data)")


#Inspect the raw data
#st.subheader('Raw data')  #rendering a dataframe as an interactive table.
#st.write(data)

#Replace the preceeding two lines with the following code to add a toggle button to show/hide the raw data table
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    

######################
#Draw a histogram

# First, add a subheader just below the raw data section:
st.subheader('Number of pickups by hour')

# Use NumPy to generate a histogram that breaks down pickup times binned by hour:

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

#Now, let's use Streamlit's st.bar_chart() method to draw this histogram.
st.bar_chart(hist_values)    #note that streamlit also supports other libraries like Matplotlib
############################

#####################################
#Plot data on a map for pick locations

#Add a subheader for the section:
st.subheader('Map of all pickups')

#Use the st.map() function to plot the data:
st.map(data)    #For complex data maps, use st.pydeck_chart

#Show the concentration of pickups at 17:00.
hour_to_filter = 17
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)
########################################

##############################
#Use a button to toggle data
#use checkbox to show/hide the raw data table at the top of your app.
# This is done in the begining of the code
############################

############################
#Share your app
#After you’ve built a Streamlit app, it's time to share it! To show it off to the world you can use Streamlit Community Cloud to deploy, manage, and share your app for free.

#It works in 3 simple steps:

#Put your app in a public GitHub repo (and make sure it has a requirements.txt!)
#Sign into share.streamlit.io
#Click 'Deploy an app' and then paste in your GitHub URL
############################



