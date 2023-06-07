import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
from math import sqrt, cos, radians

st.header(" CabiAnyone App")
st.text_input("Enter your username: ", key="name")
data = pd.read_csv("filtered_raw.csv")


# load model fare
xreg_fare = xgb.XGBRegressor()
xreg_fare.load_model("xgb_fare.json")

# load model duration
xreg_duration = xgb.XGBRegressor()
xreg_duration.load_model("xgb_duration.json")


#Features Zones
st.subheader("Please, select your trip info")
continent2 = st.sidebar.selectbox(label = "Where do you want to go today?", options = data["DOZone"])
continent = st.sidebar.selectbox(label = "Starting from?", options = data["PUZone"])

lat2 = data[data["DOZone"] == continent2].DOLat.mean()
lon2 = data[data["DOZone"] == continent2].DOLong.mean()
lat1 = data[data["PUZone"] == continent].PULat.mean()
lon1 = data[data["PUZone"] == continent].PULong.mean()

#Estimate trip distance using lat and long
R = 6371  # radius of the earth in km
x = (radians(lon2) - radians(lon1)) * cos(0.5 * (radians(lat2) + radians(lat1)))
y = radians(lat2) - radians(lat1)
trip_distance = R * sqrt(x*x + y*y)

#Feature hour
input_hour = st.slider('What hour is it now?', 0, max(data["hour_pickup"]), 1)
#Estimate median speed at that given time
speed_minutes = data[data["hour_pickup"] == input_hour]["speed_minutes"].median()

#Prediction
if st.button('Estimate my trip budget'):
    inputs1 = np.expand_dims([trip_distance, input_hour], 0)
    fare_amount = xreg_fare.predict(inputs1)
    fare_amount = float(fare_amount)
    inputs2 = np.expand_dims(
        [trip_distance, input_hour, fare_amount], 0)
    duration = xreg_duration.predict(inputs2)
    duration = float(duration)
    print("final pred", [fare_amount,duration])
    st.write(f"Your trip will cost {round(fare_amount, 2)} bucks and it will take {round(duration, 2)} minutes")
