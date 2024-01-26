import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle 
import joblib
import pandas as pd 

with open('mapping_dict1.pkl', 'rb') as file:
    mapping_dict_channelGrouping = pickle.load(file)

with open('mapping_dict2.pkl', 'rb') as file:
    mapping_dict_device_browser = pickle.load(file)

with open('mapping_dict3.pkl', 'rb') as file:
    mapping_dict_device_operatingSystem = pickle.load(file)

with open('mapping_dict4.pkl', 'rb') as file:
    mapping_dict_Products = pickle.load(file)

with open('mapping_dict5.pkl', 'rb') as file:
    mapping_dict_region = pickle.load(file)

scaler = joblib.load('standard_scaler.pkl')

model = joblib.load('dtree.pkl')

tab1, tab2, tab3 = st.tabs(['Home', 'Predict', 'About'])

with tab2:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap='small')

    with col1:
        count_session = st.text_input("Enter Session Count")
    with col2:
        count_hit = st.text_input("Enter hit Count")
    with col3:
        num_interactions = st.text_input("Enter number of interactions")
    with col4:
        channelGrouping = st.selectbox('Channel Grouping', options=list(mapping_dict_channelGrouping.keys()))
    
    col5, col6, col7, col8 = st.columns([1, 1, 1, 1], gap='small')

    with col5:
        device_browser = st.selectbox('Device Browser', options=list(mapping_dict_device_browser.keys()))
    with col6:
        device_operatingSystem = st.selectbox('Operating System', options=list(mapping_dict_device_operatingSystem.keys()))
    with col7:
       Products = st.selectbox('Select Products', options=list(mapping_dict_Products.keys()))
    with col8:
        region = st.selectbox('Select Region', options=list(mapping_dict_region.keys()))

# Create DataFrame after user inputs
df = pd.DataFrame({
    'count_session': [float(count_session)],
    'count_hit': [float(count_hit)],
    'num_interactions': [float(num_interactions)],
    'channelGrouping': [channelGrouping],
    'device_browser': [device_browser],
    'device_operatingSystem': [device_operatingSystem],
    'Products': [Products],
    'region': [region]})

df_encoded = df.copy()
df_encoded['channelGrouping'] = df['channelGrouping'].map(mapping_dict_channelGrouping)
df_encoded['device_browser'] = df['device_browser'].map(mapping_dict_device_browser)
df_encoded['device_operatingSystem'] = df['device_operatingSystem'].map(mapping_dict_device_operatingSystem)
df_encoded['Products'] = df['Products'].map(mapping_dict_Products)
df_encoded['region'] = df['region'].map(mapping_dict_region)

dfc=df_encoded.drop(['count_session','count_hit','num_interactions'],axis=1)
# Select numeric columns for scaling
dfn = df_encoded.drop(['channelGrouping','device_browser','device_operatingSystem','Products','region'],axis=1)

# Scale numeric columns
dfn_scaled = pd.DataFrame(scaler.transform(dfn), columns=dfn.columns)



dff=pd.concat([dfn_scaled,dfc],axis=1)

if st.button('Predict'):

    prediction=model.predict(dff)

if prediction==1:
    st.success('Sale Converted')
else:
    st.success('Sale Not Converted')

