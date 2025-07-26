import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="EV Charging Forecast", layout="wide")

st.title("🔋 EV Home-Charging Cost & Consumption Analysis")
st.markdown("""
This app analyzes and forecasts monthly energy usage from home EV charging data using **Facebook Prophet**.
""")

DATA_PATH = "dataset/Dataset1_charging_reports.csv"

try:
    df = pd.read_csv(DATA_PATH, sep=";")
except Exception as e:
    st.error(f"❌ Failed to load dataset from {DATA_PATH}: {e}")
    st.stop()

if df.columns[0] == 'location;user_id;session_id;plugin_time;plugout_time;connection_time;energy_session':
    df = df.iloc[1:]
    df.columns = ['location', 'user_id', 'session_id', 'plugin_time', 'plugout_time', 'connection_time', 'energy_session']
else:
    df.columns = [col.strip().lower() for col in df.columns]

# ---- Preprocessing ----
df['plugin_time'] = pd.to_datetime(df['plugin_time'])
df['plugout_time'] = pd.to_datetime(df['plugout_time'])
df['energy_session'] = pd.to_numeric(df['energy_session'], errors='coerce')

df['month'] = df['plugin_time'].dt.to_period('M').dt.to_timestamp()
df_monthly = df.groupby('month')['energy_session'].sum().reset_index()
df_monthly.columns = ['ds', 'y']

# ---- Forecasting ----
m = Prophet()
m.fit(df_monthly)

future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# ---- Evaluation ----
merged_df = forecast.set_index('ds')[['yhat']].join(df_monthly.set_index('ds')[['y']]).dropna()
mae = mean_absolute_error(merged_df['y'], merged_df['yhat'])
rmse = np.sqrt(mean_squared_error(merged_df['y'], merged_df['yhat']))

# ---- Visualization ----
st.subheader("📊 Monthly Forecast vs Actual Usage")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(merged_df.index, merged_df['y'], label='Actual', marker='o')
ax1.plot(merged_df.index, merged_df['yhat'], label='Predicted', marker='x')
ax1.set_title("Monthly Energy Usage Forecast")
ax1.set_ylabel("Energy (kWh)")
ax1.legend()
st.pyplot(fig1)

# ---- Cost Analysis ----
COST_PER_KWH = 8
avg_actual_cost = merged_df['y'].mean() * COST_PER_KWH
avg_pred_cost = merged_df['yhat'].mean() * COST_PER_KWH

st.subheader("📈 Forecast Summary")
st.markdown(f"- **Mean Absolute Error (MAE):** {mae:.2f} kWh")
st.markdown(f"- **Root Mean Square Error (RMSE):** {rmse:.2f} kWh")
st.markdown(f"- **Average Actual Monthly Cost:** ₹{avg_actual_cost:.2f}")
st.markdown(f"- **Average Predicted Monthly Cost:** ₹{avg_pred_cost:.2f}")

# ---- Future Forecast ----
st.subheader("🔮 Forecast for Next 12 Months")
forecast_tail = forecast[['ds', 'yhat']].tail(12).copy()
forecast_tail['predicted_cost'] = forecast_tail['yhat'] * COST_PER_KWH
st.dataframe(forecast_tail.rename(columns={
    'ds': 'Month',
    'yhat': 'Predicted Energy (kWh)',
    'predicted_cost': 'Estimated Cost (₹)'
}))
