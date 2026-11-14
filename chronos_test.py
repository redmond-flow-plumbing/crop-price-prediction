"""
Chronos zero-shot test — wheat prices (runs after torch + chronos install).
Same test window as the main script: last 12 months (Oct 2025 - Sep 2026).
Prints Chronos MAE / RMSE / MAPE so it can be compared with
Seasonal Naive (44.57 RMSE), Linear Regression (15.11) and XGBoost (14.81).

Run:  python3 chronos_test.py
"""

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

torch.manual_seed(42)  # Chronos samples, so fix the seed for reproducible numbers

DATA = "data/PWHEAMTUSDM.csv"
TEST_MONTHS = 12

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["observation_date"])
df["price"] = pd.to_numeric(df["PWHEAMTUSDM"], errors="coerce")
df = df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)

prices = df["price"].values.astype(np.float32)
actual_test = prices[-TEST_MONTHS:]
context = torch.tensor(prices[:-TEST_MONTHS])

print(f"Context: {len(context)} months to {df['date'].iloc[-TEST_MONTHS-1].date()}")
print("Loading amazon/chronos-t5-small (first run downloads ~200 MB)...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32
)

print(f"Forecasting next {TEST_MONTHS} months...")
forecast = pipeline.predict(context, prediction_length=TEST_MONTHS, num_samples=20)
pred = np.median(forecast.numpy(), axis=1).reshape(-1).astype(np.float64)  # median over the 20 samples

mae = float(np.mean(np.abs(actual_test - pred)))
rmse = float(np.sqrt(np.mean((actual_test - pred) ** 2)))
mape = float(np.mean(np.abs((actual_test - pred) / actual_test)) * 100)
print(f"\nChronos (t5-small, zero-shot) — test MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
print("Compare: Seasonal Naive RMSE 44.57 | Linear Regression 15.11 | XGBoost 14.81")

out = pd.DataFrame({
    "date": df["date"].iloc[-TEST_MONTHS:].values,
    "actual": actual_test,
    "chronos": np.round(pred, 2),
})
out.to_csv("predictions_wheat_chronos.csv", index=False)
print("Saved: predictions_wheat_chronos.csv")

# 6-month forecast from the full history
full_context = torch.tensor(prices)
forecast6 = pipeline.predict(full_context, prediction_length=6, num_samples=20)
pred6 = np.median(forecast6.numpy(), axis=1).reshape(-1).astype(np.float64)
dates6 = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")
fc = pd.DataFrame({"date": dates6, "forecast_price": np.round(pred6, 2)})
fc.to_csv("forecast_next6_wheat_chronos.csv", index=False)
print("\nNext 6 months (Chronos):")
print(fc.to_string(index=False))
print("Saved: forecast_next6_wheat_chronos.csv")
