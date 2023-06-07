import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
import geopy.distance
import joblib

st.set_page_config(
    page_title="Anyone IApp",
    page_icon=":robot:"
)
st.header("Anyone IApp")
st.subheader("_Travel safer and smarter_')        
         
data = pd.read_csv("filtered_raw.csv")

#load fare scaler
fare_scaler = joblib.load("fare_scaler")

#load duration scaler
duration_scaler = joblib.load("duration_scaler") 

# load model fare
xreg_fare = xgb.XGBRegressor()
xreg_fare.load_model("xgb_fare.json")

# load model duration
xreg_duration = xgb.XGBRegressor()
xreg_duration.load_model("xgb_duration.json")


#Features Zones
st.subheader("Please, select your trip info")
continent2 = st.sidebar.selectbox(label = "Where do you want to go today?", options = data["DOZone"].unique())
continent = st.sidebar.selectbox(label = "Starting from?", options = data["PUZone"].unique())
continent2 = str(continent2)
continent = str(continent)
lat2 = data[data["DOZone"] == continent2].DOLat.mean()
lon2 = data[data["DOZone"] == continent2].DOLong.mean()
lat1 = data[data["PUZone"] == continent].PULat.mean()
lon1 = data[data["PUZone"] == continent].PULong.mean()

#Estimate trip distance using lat and long
coords_1 = (lat1, lon1)
coords_2 = (lat2, lon2)

trip_distance = geopy.distance.geodesic(coords_1, coords_2).km

#Feature hour
input_hour = st.slider('What hour is it now?', 0, max(data["hour_pickup"]), 1)
#Estimate median speed at that given time
speed_minutes = data[data["hour_pickup"] == input_hour]["speed_minutes"].median()

#Prediction
if st.button('Estimate my trip budget'):
    inputs1 = {"trip_distance" : trip_distance, "speed_minutes": speed_minutes}
    inputs1 = pd.DataFrame([inputs1])
    inputs1 = fare_scaler.transform(inputs1)
    fare_amount = xreg_fare.predict(inputs1)
    fare_amount2 = float(fare_amount)
    inputs2 = {"trip_distance" : trip_distance, "speed_minutes": speed_minutes, "fare_amount": fare_amount2}
    inputs2 = pd.DataFrame([inputs2])
    inputs2 = duration_scaler.transform(inputs2)
    duration = xreg_duration.predict(inputs2)
    duration = float(duration)
    st.write(f"Your trip will cost {round(fare_amount2, 2)} bucks and it will take {round(duration, 2)} minutes")
