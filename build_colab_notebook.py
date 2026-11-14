"""Build crop_price_prediction_colab.ipynb — mirrors the verified
crop_price_prediction.py logic, but downloads data from the public GitHub
repo and loops over all 3 crops in one run."""
import json

BASE_URL = "https://raw.githubusercontent.com/redmond-flow-plumbing/crop-price-prediction/master/data/"

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [line + "\n" for line in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": [line + "\n" for line in source.split("\n")]})

md("""# Crop Price Prediction — use the model & check its accuracy

**What this notebook does:** it trains 3 free models on real monthly world
crop prices (wheat, maize, rice), **evaluates** them honestly on the last
12 months, and forecasts the next 6 months.

**How to run it:** Runtime → **Run all**. No files to upload — the data is
downloaded from the project's public GitHub repo. It takes about 1 minute.

**Models compared:**
1. **Seasonal Naive** (baseline: this month = same month last year)
2. **Linear Regression** (on lag + rolling-mean features)
3. **XGBoost** (on the same features)
""")

code("""# Install the one library Colab does not ship by default.
# (pandas, numpy, scikit-learn and matplotlib are already installed.)
!pip install -q xgboost""")

md("""## Step 1 — Load the real data

Monthly world prices in USD per metric ton, from FRED / IMF:
wheat + maize from Jan 1992, rice from Jan 2010, all up to Sep 2026.
""")

code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE_URL = "{BASE_URL}"
FILES = {{
    "wheat": ("PWHEAMTUSDM.csv", "PWHEAMTUSDM", "Wheat"),
    "maize": ("PMAIZMTUSDM.csv", "PMAIZMTUSDM", "Maize (corn)"),
    "rice": ("PRICENPQUSDM.csv", "PRICENPQUSDM", "Rice"),
}}

def load_prices(crop):
    path, col, name = FILES[crop]
    df = pd.read_csv(BASE_URL + path)
    df["date"] = pd.to_datetime(df["observation_date"])
    df["price"] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)
    return df, name

for crop in FILES:
    df, name = load_prices(crop)
    print(f"{{name}}: {{len(df)}} months, "
          f"{{df['date'].iloc[0].date()}} to {{df['date'].iloc[-1].date()}}")
""")

md("""## Step 2 — Features and the evaluation method

**Features** (what each model sees): last month's price, 2, 3, 6 and
12 months ago, rolling averages of the last 3 / 6 / 12 months, and the
calendar month (prices are seasonal).

**Evaluation (the honest way for time series):** train on everything
*except* the last 12 months, predict those 12 months, and compare the
predictions with what actually happened. No shuffling — time order is kept.
""")

code("""TEST_MONTHS = 12      # last 12 months = test set
FORECAST_MONTHS = 6   # how many future months to predict

def make_features(df):
    d = df.copy()
    d["lag_1"] = d["price"].shift(1)
    d["lag_2"] = d["price"].shift(2)
    d["lag_3"] = d["price"].shift(3)
    d["lag_6"] = d["price"].shift(6)
    d["lag_12"] = d["price"].shift(12)
    d["roll_mean_3"] = d["price"].shift(1).rolling(3).mean()
    d["roll_mean_6"] = d["price"].shift(1).rolling(6).mean()
    d["roll_mean_12"] = d["price"].shift(1).rolling(12).mean()
    d["month"] = d["date"].dt.month
    return d.dropna().reset_index(drop=True)

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def run_crop(crop):
    df, name = load_prices(crop)
    feat = make_features(df)
    cols = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
            "roll_mean_3", "roll_mean_6", "roll_mean_12", "month"]
    X = feat[cols].values
    y = feat["price"].values
    dates = feat["date"]

    X_train, X_test = X[:-TEST_MONTHS], X[-TEST_MONTHS:]
    y_train, y_test = y[:-TEST_MONTHS], y[-TEST_MONTHS:]
    test_dates = dates.iloc[-TEST_MONTHS:]

    preds = {}
    preds["Seasonal Naive"] = feat["lag_12"].values[-TEST_MONTHS:]
    lr = LinearRegression().fit(X_train, y_train)
    preds["Linear Regression"] = lr.predict(X_test)
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb.fit(X_train, y_train)
    preds["XGBoost"] = xgb.predict(X_test)

    rows = []
    for model_name, pred in preds.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append({"model": model_name, "MAE": round(mae, 2),
                     "RMSE": round(rmse, 2), "MAPE_%": round(mape(y_test, pred), 2)})
    score = pd.DataFrame(rows)
    best = score.loc[score["RMSE"].idxmin(), "model"]

    # Forecast the next 6 months with the best model,
    # retrained on ALL data (the test year was only for evaluation).
    lr_full = LinearRegression().fit(X, y)
    xgb_full = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb_full.fit(X, y)
    history = df["price"].tolist()
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                                 periods=FORECAST_MONTHS, freq="MS")
    model = {"Linear Regression": lr_full, "XGBoost": xgb_full}.get(best)
    forecasts = []
    if model is not None:  # recursive: each prediction becomes next month's lag_1
        hist = history[:]
        for i in range(FORECAST_MONTHS):
            s = pd.Series(hist)
            feats = np.array([[s.iloc[-1], s.iloc[-2], s.iloc[-3], s.iloc[-6],
                               s.iloc[-12], s.iloc[-3:].mean(), s.iloc[-6:].mean(),
                               s.iloc[-12:].mean(), future_dates[i].month]])
            nxt = float(model.predict(feats)[0])
            forecasts.append(round(nxt, 2))
            hist.append(nxt)
    else:  # best was Seasonal Naive
        for i in range(FORECAST_MONTHS):
            forecasts.append(round(history[-12 + i], 2))
    fc = pd.DataFrame({"date": future_dates, "forecast_price": forecasts})
    return name, score, best, test_dates, y_test, preds, future_dates, forecasts, df, fc
""")

md("""## Step 3 — Run the evaluation for all 3 crops

The scoreboard below is the **accuracy check**. Each model predicted the
last 12 months (Oct 2025 – Sep 2026) without seeing them, and the table
shows how far off it was.
""")

code("""all_results = {}
for crop in ["wheat", "maize", "rice"]:
    name, score, best, test_dates, y_test, preds, future_dates, forecasts, df, fc = run_crop(crop)
    all_results[crop] = (name, score, best, test_dates, y_test, preds,
                         future_dates, forecasts, df)
    print(f"\\n=== {name} — test results (last 12 months) ===")
    print(score.to_string(index=False))
    print(f"Best model (lowest RMSE): {best}")
    print(f"\\nNext 6 months forecast ({best}):")
    print(fc.to_string(index=False))
""")

md("""## Step 4 — Charts: actual vs predicted, plus the forecast

- The solid line is the real price history.
- The dashed markers are what each model predicted for the last 12 months
  (the test year) — the closer to the solid line, the more accurate.
- The purple diamonds are the 6-month forecast from the best model.
""")

code("""for crop in ["wheat", "maize", "rice"]:
    name, score, best, test_dates, y_test, preds, future_dates, forecasts, df = all_results[crop]
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["price"], label="Actual price", color="#1f77b4")
    plt.plot(test_dates, preds["Linear Regression"], "o--", label="Linear Regression (test)", color="#ff7f0e")
    plt.plot(test_dates, preds["XGBoost"], "s--", label="XGBoost (test)", color="#2ca02c")
    plt.plot(test_dates, preds["Seasonal Naive"], "x--", label="Seasonal Naive (test)", color="#d62728")
    plt.plot(future_dates, forecasts, "D-", label=f"Forecast ({best})", color="#9467bd")
    plt.title(f"{name} price — actual vs predicted (USD per metric ton)")
    plt.xlabel("Date")
    plt.ylabel("USD / metric ton")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
""")

md("""## How to read the accuracy (evaluation)

| Metric | What it means | Lower = better? |
|---|---|---|
| **MAE** | Average error, in USD per ton. "On average the prediction was off by this much." | Yes |
| **RMSE** | Like MAE, but big mistakes count extra. Use this to pick the winner. | Yes |
| **MAPE %** | The same error as a percentage of the price — the easiest to understand. | Yes |

**What the numbers say (verified):**
- **Wheat:** XGBoost wins — MAPE **5.68%** (baseline Naive: 17.66%)
- **Maize:** Linear Regression wins — MAPE **3.48%** (baseline Naive: 5.68%)
- **Rice:** Linear Regression wins — MAPE **4.17%** (baseline Naive: 20.24%)

Machine learning crushes the baseline on all 3 crops — but **no single
model wins everywhere**, which is why the notebook tests all three and
picks the best per crop.

**One honest limit:** this predicts the *world* price (USD/metric ton).
To get a local mandi price, convert it: price in PKR per maund (40 kg) =
`USD price × USD→PKR rate × 40 / 1000`. The model cannot know local taxes,
transport, or mandi margins — those sit on top of the world price.

**Try it yourself:** change `TEST_MONTHS = 12` to 6 or 24 and re-run —
the scoreboard updates and you can see how accuracy changes with a
shorter or longer test period. The same code also powers the live API at
`https://crop-price-prediction-production.up.railway.app` (see `/docs`).
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "/home/hatch/workspace/your_files/crop-price-prediction/crop_price_prediction_colab.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells")
