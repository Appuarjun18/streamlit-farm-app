
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Simulated training data and model (replace with your actual data/model loading)
total_growth_days = 121

# Generate dummy data
data = []
for day_num in range(1, total_growth_days + 1):
    temp = np.random.uniform(18, 30)
    humidity = np.random.uniform(50, 90)
    soil_pH = 6.5
    organic_matter = 3.2
    nitrogen = 25
    growth_progress = day_num
    yield_kg = (growth_progress / total_growth_days) * 8000 + np.random.normal(0, 100)
    data.append([temp, humidity, soil_pH, organic_matter, nitrogen, growth_progress, yield_kg])

columns = ['temperature', 'humidity', 'soil_pH', 'organic_matter', 'nitrogen', 'growth_progress_days', 'yield_kg']
df = pd.DataFrame(data, columns=columns)

X = df.drop('yield_kg', axis=1)
y = df['yield_kg']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def predict_yield_scenario(input_features):
    df_input = pd.DataFrame([input_features], columns=X.columns)
    return model.predict(df_input)[0]

st.title("Digital Twin for Farm Crop Yield Prediction")

st.sidebar.header("Adjust Environmental Parameters")
temperature = st.sidebar.slider("Temperature (°C)", 10, 40, 28)
humidity = st.sidebar.slider("Humidity (%)", 30, 100, 75)
soil_pH = st.sidebar.slider("Soil pH", 4.0, 9.0, 6.5)
organic_matter = st.sidebar.slider("Organic Matter (%)", 0.0, 10.0, 3.2)
nitrogen = st.sidebar.slider("Soil Nitrogen (mg/kg)", 0, 100, 25)
growth_days = st.sidebar.slider("Growth Progress (days)", 0, total_growth_days, 20)

input_features = {
    'temperature': temperature,
    'humidity': humidity,
    'soil_pH': soil_pH,
    'organic_matter': organic_matter,
    'nitrogen': nitrogen,
    'growth_progress_days': growth_days
}

pred_yield = predict_yield_scenario(input_features)

st.write(f"### Predicted Crop Yield: {pred_yield:.2f} kg/ha")
