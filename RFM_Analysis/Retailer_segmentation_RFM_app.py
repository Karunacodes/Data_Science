# -*- coding: utf-8 -*-
"""Streamlit_Customer-Segmentation App

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rRHFDGsPjhJlKvWgyd4IcWWGdovhIATJ
"""
pip install pycaret


"""# **Streamlit Deployment**"""

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = loaded_model
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

with open(fname, 'rt') as f:
    lines = [x.strip() for x in f.readlines()]

import pickle

loaded_model = pickle.load(open("/content/drive/MyDrive/DS Projects/Project 72/Datasets/final_dt.pkl",'rb'))

st.title("Customer-Segmentation based on RFM Scores for Pharma Retailers")
st.markdown('The dashboard will visualize the Customer-Segments of Pharma Retailers')
st.markdown('**RFM (Recency-Frequency-Monetary) analysis** is a simple technique for behaviour based customer segmentation')
st.sidebar.title("Visualization Selector")
st.sidebar.markdown("Select the Charts/Plots accordingly:")

select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
if not st.sidebar.checkbox("Hide", True, key='1'):
     if select == 'Pie chart':
                         st.title("Top 5 Retailers")
                         fig=px.pie(df, values=df_rfm['retailer_names'][:5], names=df_rfm['RFM_Score'][:5], title='Top 5 Retailers')
                         st.plotly_chart(fig)
                         
                         if select=='Bar plot':
                           st.title("Selected Top 5 Retailers")
                           fig = go.Figure(data=[
        go.Bar(name='Recency', x=df_rfm['reailers_names'][:5], y=df['Recency'][:5]),
        go.Bar(name='Frequency', x=df_rfm['retailer_names'][:5], y=df['Frequency'][:5]),
        go.Bar(name='Monetary', x=df_rfm['retailer_names'][:5], y=df['Monetary'][:5])])
                           st.plotly_chart(fig)

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def main():
    from PIL import Image
    image = Image.open('Customer Lifetime Value.png')
    image_pharmacy = Image.open('Pharmacy.jpg')
    st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict if an employee will leave the company')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_pharmacy)
    st.title("Predicting Pharma Retailers Segment")
    
    if add_selectbox == 'Online':
      master_order_id =st.text_input("Enter Master Order Id")
      retailer_names =st.text_input('Enter Retailer Name')
      RFM_Score = st.text_input('RFM_Score')
      Customer_Segment=""
      input_dict={'master_order_id':master_order_id, 'retailer_names':retailer_names,'RFM_Score':RFM_Score,'Customer_Segment':Customer_Segment}
      input_df = pd.DataFrame([input_dict])
      
      if st.button("Predict"):
        Customer_Segment = predict(model=model, input_df=input_df)
        Customer_Segment = str(Customer_Segment)
        st.success('The output is {}'.format(Customer_Segment))

      if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
