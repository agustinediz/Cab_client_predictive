import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header(" Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("filtered_df.csv")


encoder_PU = LabelEncoder()
encoder_PU.classes_ = np.load('classes_PU.npy',allow_pickle=True)

encoder_DO = LabelEncoder()
encoder_DO.classes_ = np.load('classes_DO.npy',allow_pickle=True)


# load model
xreg_duration = xgb.XGBRegressor()
xreg_duration.load_model("xgb_duration.json")


st.subheader("Please select relevant features of your trip!")
left_column, right_column = st.columns(2)
with left_column:
    inp_species = st.radio(
        'Where are you?:',
        np.unique(data['PUZone']))


input_hour = st.slider('What hour it is?', 0.0, max(data["hour_pickup"]), 1.0)


if st.button('Make Prediction'):
    input_PU = encoder_PU.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [int(input_PU), input_hour], 0)
    prediction = xreg_duration.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your trip duration is: {np.squeeze(prediction, -1)} minutes")
