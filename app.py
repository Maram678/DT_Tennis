import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Play Tennis?", page_icon="🎾")
st.title("🎾 Should we play tennis?")
st.write("A decision tree trained on the tennis dataset.")

@st.cache_resource              # load the model only once
def load_model():
    return joblib.load("tennis_model.joblib")

model = load_model()

outlook = st.selectbox("Outlook", ["Sunny", "Overcast", "Rainy"], index=1)
temperature = st.selectbox("Temperature", ["Hot", "Mild", "Cool"])
humidity = st.radio("Humidity", ["High", "Normal"], horizontal=True)
windy = st.checkbox("Windy")

day = pd.DataFrame([{"Outlook": outlook, "Temperature": temperature,
                     "Humidity": humidity, "Windy": windy}])
answer = model.predict(day)[0] 
confidence = model.predict_proba(day).max() 

if answer == "Yes":
    st.success(f"Yes, let's play! (confidence {confidence:.0%})")
else:
    st.error(f"No, better stay home. (confidence {confidence:.0%})")
