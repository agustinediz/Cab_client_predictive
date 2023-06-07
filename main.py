import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
from math import sqrt, cos, radians

st.header(" Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("filtered_raw.csv")


# load model fare
xreg_fare = xgb.XGBRegressor()
xreg_fare.load_model("xgb_fare.json")

# load model duration
xreg_duration = xgb.XGBRegressor()
xreg_duration.load_model("xgb_duration.json")


#Features Zones
#metric_labels = {"PUZone": "Zone of Pickup", "DOZone": "Drop off Zone"}

#def format_metric(metric_raw):
    #return metric_labels[metric_raw]

st.subheader("Please select your trip!")
continent = st.sidebar.selectbox(label = "Starting from?", options = data["PUZone"])
continent2 = st.sidebar.selectbox(label = "Where do you want to go today?", options = data["DOZone"])

lat2 = data[data["DOZone"] == continent2].DOLat.mean()
lon2 = data[data["DOZone"] == continent2].DOLong.mean()
lat1 = data[data["PUZone"] == continent].PULat.mean()
lon1 = data[data["PUZone"] == continent].PULong.mean()

R = 6371  # radius of the earth in km
x = (radians(lon2) - radians(lon1)) * cos(0.5 * (radians(lat2) + radians(lat1)))
y = radians(lat2) - radians(lat1)
trip_distance = R * sqrt(x*x + y*y)

#Feature hour
input_hour = st.slider('What hour is it now?', 0, max(data["hour_pickup"]), 1)


speed_minutes = data[data["hour_pickup"] == input_hour]["speed_minutes"].median()

#Prediction
if st.button('Make Prediction'):
    inputs1 = np.expand_dims([trip_distance, input_hour], 0)
    fare_amount = xreg_fare.predict(inputs)
    inputs2 = np.expand_dims(
        [trip_distance, input_hour, fare_amount], 0)
    duration = xreg_duration.predict(inputs2)
    print("final pred", np.squeeze([fare_amount,duration], -1))
    st.write(f"Your trip will cost {np.squeeze(fare_amount, -1)} and the trip duration is {np.squeeze(duration, -1)} minutes")
