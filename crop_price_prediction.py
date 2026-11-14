"""
Crop Price Prediction — working starter project
================================================
Data: real monthly world prices (USD per metric ton) from FRED / IMF,
downloaded as CSV. Default crop is WHEAT (Jan 1992 - present).
Change CROP below to "maize" or "rice" to run the same pipeline on them.

Models compared (all free, all in this one file):
  1. Seasonal Naive  (baseline: this month = same month last year)
  2. Linear Regression (on lag + rolling-mean features)
  3. XGBoost          (on the same features)

Train/test split: last 12 months are the test set (time order kept,
no shuffling — that is the correct way for time series).
Metrics: MAE, RMSE, MAPE. The best model then forecasts the next 6 months.

Run:  python3 crop_price_prediction.py
Needs: pandas, numpy, scikit-learn, xgboost, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed, saves PNG files
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
CROP = "wheat"  # change to "maize" or "rice"
FILES = {
    "wheat": ("data/PWHEAMTUSDM.csv", "PWHEAMTUSDM", "Wheat"),
    "maize": ("data/PMAIZMTUSDM.csv", "PMAIZMTUSDM", "Maize (corn)"),
    "rice": ("data/PRICENPQUSDM.csv", "PRICENPQUSDM", "Rice"),
}
TEST_MONTHS = 12      # last 12 months = test set
FORECAST_MONTHS = 6   # how many future months to predict


def load_prices(crop):
    path, col, name = FILES[crop]
    df = pd.read_csv(os.path.join(BASE, path))
    df["date"] = pd.to_datetime(df["observation_date"])
    df["price"] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)
    return df, name


def make_features(df):
    """Lag + rolling-mean features. Target = this month's price."""
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
    d = d.dropna().reset_index(drop=True)
    return d


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main():
    df, name = load_prices(CROP)
    print(f"Crop: {name} — {len(df)} months, "
          f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    print(df["price"].describe().round(2).to_string())

    feat = make_features(df)
    feature_cols = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
                    "roll_mean_3", "roll_mean_6", "roll_mean_12", "month"]
    X = feat[feature_cols].values
    y = feat["price"].values
    dates = feat["date"]

    # --- time-ordered split: last 12 months = test ---
    X_train, X_test = X[:-TEST_MONTHS], X[-TEST_MONTHS:]
    y_train, y_test = y[:-TEST_MONTHS], y[-TEST_MONTHS:]
    test_dates = dates.iloc[-TEST_MONTHS:]
    print(f"\nTrain: {len(X_train)} months, Test: {len(X_test)} months "
          f"({test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()})")

    results = {}

    # 1) Seasonal Naive: forecast = price 12 months ago
    pred_naive = feat["lag_12"].values[-TEST_MONTHS:]
    results["Seasonal Naive"] = pred_naive

    # 2) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["Linear Regression"] = lr.predict(X_test)

    # 3) XGBoost
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb.fit(X_train, y_train)
    results["XGBoost"] = xgb.predict(X_test)

    # --- scoreboard ---
    print("\n=== Test results (last 12 months) ===")
    rows = []
    for model_name, pred in results.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append((model_name, mae, rmse, mape(y_test, pred)))
        print(f"{model_name:20s} MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape(y_test, pred):5.2f}%")
    score = pd.DataFrame(rows, columns=["model", "MAE", "RMSE", "MAPE_%"])
    best = score.loc[score["RMSE"].idxmin(), "model"]
    print(f"\nBest model (lowest RMSE): {best}")

    # --- save test predictions ---
    out = pd.DataFrame({
        "date": test_dates.values,
        "actual": y_test,
        "seasonal_naive": results["Seasonal Naive"],
        "linear_regression": results["Linear Regression"],
        "xgboost": results["XGBoost"],
    })
    out_path = os.path.join(BASE, f"predictions_{CROP}.csv")
    out.to_csv(out_path, index=False)
    print(f"Test predictions saved: {out_path}")

    # --- forecast the next 6 months with the best model ---
    # Retrain on ALL data first: the test set above was only for evaluation.
    # The live forecast should use every month available, including the
    # last 12 months that were held out for testing.
    lr_full = LinearRegression().fit(X, y)
    xgb_full = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb_full.fit(X, y)
    print(f"(Forecast model retrained on all {len(X)} months, including the test year.)")
    # Recursive: each new prediction becomes next month's lag_1.
    history = df["price"].tolist()
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                                 periods=FORECAST_MONTHS, freq="MS")
    model = {"Linear Regression": lr_full, "XGBoost": xgb_full}.get(best)
    forecasts = []
    if model is not None:
        hist = history[:]
        for _ in range(FORECAST_MONTHS):
            s = pd.Series(hist)
            feats = np.array([[s.iloc[-1], s.iloc[-2], s.iloc[-3], s.iloc[-6],
                               s.iloc[-12], s.iloc[-3:].mean(), s.iloc[-6:].mean(),
                               s.iloc[-12:].mean(),
                               (future_dates[len(forecasts)].month)]])
            nxt = float(model.predict(feats)[0])
            forecasts.append(nxt)
            hist.append(nxt)
    else:  # best was Seasonal Naive
        for i in range(FORECAST_MONTHS):
            forecasts.append(history[-12 + i])
    fc = pd.DataFrame({"date": future_dates, "forecast_price": np.round(forecasts, 2)})
    fc_path = os.path.join(BASE, f"forecast_next6_{CROP}.csv")
    fc.to_csv(fc_path, index=False)
    print(f"\nNext {FORECAST_MONTHS} months forecast ({best}):")
    print(fc.to_string(index=False))
    print(f"Forecast saved: {fc_path}")

    # --- chart: history + test predictions + forecast ---
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["price"], label="Actual price", color="#1f77b4")
    plt.plot(test_dates, results["Linear Regression"], "o--", label="Linear Regression (test)", color="#ff7f0e")
    plt.plot(test_dates, results["XGBoost"], "s--", label="XGBoost (test)", color="#2ca02c")
    plt.plot(test_dates, results["Seasonal Naive"], "x--", label="Seasonal Naive (test)", color="#d62728")
    plt.plot(future_dates, forecasts, "D-", label=f"Forecast ({best})", color="#9467bd")
    plt.title(f"{name} price — actual vs predicted (USD per metric ton)")
    plt.xlabel("Date")
    plt.ylabel("USD / metric ton")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(BASE, f"chart_{CROP}.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved: {chart_path}")


if __name__ == "__main__":
    main()
