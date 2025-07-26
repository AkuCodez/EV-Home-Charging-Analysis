# 🔋 EV Home-Charging Cost & Consumption Analysis

This project explores and forecasts electric vehicle (EV) charging behavior at home using a real-world dataset. It analyzes charging session patterns, energy usage, and costs, and builds a predictive model using Facebook Prophet to forecast monthly energy consumption and cost.

---

## 📌 Objectives

- Analyze home EV charging session data: plugin times, durations, energy consumed.
- Understand user behavior and identify peak usage trends.
- Forecast monthly energy consumption and cost using **time series modeling**.
- Evaluate model performance using error metrics (MAE, RMSE).
- Visualize actual vs predicted energy usage and cost over time.

---

## 🗂️ Dataset Overview

The dataset contains the following columns:

- `location` – Charging station location
- `user_id`, `session_id` – Unique identifiers
- `plugin_time`, `plugout_time` – Timestamps of start and end of each session
- `connection_time` – Duration (in seconds)
- `energy_session` – Energy consumed during session (in kWh)

> Total Records: ~50,000+  
> Time Range: Feb 2018 to Dec 2021 (monthly aggregation)

---

## 🛠️ Tools & Technologies

- **Python** – Pandas, NumPy, Matplotlib, Seaborn
- **Prophet** – Time Series Forecasting
- **Google Colab** – Development environment

---

## 📈 Project Highlights

### 1. Data Cleaning & Preprocessing
- Converted time columns to `datetime` objects.
- Filtered incomplete records.
- Extracted hourly and daily usage trends.

### 2. Feature Engineering
- Calculated `session_duration_hrs` and `energy_per_hour`.
- Resampled data to monthly aggregation for forecasting.

### 3. Forecasting
- Used Facebook Prophet for univariate time series forecasting of monthly energy usage.
- Trained on historical data to predict future energy trends.

### 4. Evaluation
- **MAE (Mean Absolute Error):** 77.88 kWh  
- **RMSE (Root Mean Square Error):** 91.27 kWh  
- Aligned forecast with actuals using `MonthEnd` adjustment to ensure accurate merge.

### 5. Cost Analysis
- Assumed electricity rate: ₹10 per kWh
- Calculated monthly costs (actual vs predicted)
- Average Actual Cost: ₹796.25  
- Average Predicted Cost: ₹785.16  

---

## 📊 Visualizations

- Actual vs Predicted Monthly Energy Usage
- Monthly Cost Comparison
- Distribution of Charging Session Durations
- Hourly Charging Behavior Heatmap

---

## 📍 Use Case

This project is tailored to demonstrate:
- Time Series Forecasting
- Real-world energy data analysis
- Predictive cost modeling
---

## 🚀 How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Upload the dataset (`ev_charging_data.csv` or similar).
3. Run each cell step-by-step.
4. Visualizations and forecasts will appear inline.

---

## 🙋‍♂️ Author

**Akshaj**  
[GitHub: AkuCodez](https://github.com/AkuCodez)  
📫 Reach out for collaboration or feedback!

---
