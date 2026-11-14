"""
Daily crop price test — 7-day and 30-day forecasts WITH honest evaluation
=========================================================================
Data: daily futures closes from Yahoo Finance (free).
  Wheat ZW=F and Corn ZC=F, Nov 2021 - present, converted to USD/metric ton:
  wheat 1 bushel = 27.2155 kg  ->  x36.7437 ;  corn 1 bushel = 25.4012 kg -> x39.3680

Why a backtest? A forecast for the future has no actuals yet, so its error
cannot be measured. Instead we rewind: pick 6 recent dates (origins), train
only on data up to each origin, forecast 7 and 30 trading days ahead, and
score against what actually happened. The average error across origins is
the honest accuracy number for each horizon.

Models: Naive (last price carried forward), Linear Regression, XGBoost,
Chronos (t5-small, zero-shot, median of 20 samples, seed 42).
LR/XGBoost forecast recursively (each predicted day feeds the next day's lags).
Chronos forecasts all steps directly.

Trading days: futures trade Mon-Fri (US holidays aside), so "7 days" means
7 trading days (~9-10 calendar days) and "30 days" means 30 trading days
(~6 calendar weeks). Forecast dates use a business-day calendar.

Run:  python3 daily_forecast.py        (takes a few minutes, Chronos is slow on CPU)
Needs: pandas, numpy, scikit-learn, xgboost, matplotlib, torch, chronos
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
CROPS = {
    "wheat": ("data/daily_zw.csv", 36.7437, "Wheat"),
    "corn": ("data/daily_zc.csv", 39.3680, "Corn"),
}
HORIZONS = [7, 30]
N_ORIGINS = 6          # backtest origins (spaced 30 trading days apart)
CHRONOS_ORIGINS = 3    # Chronos is slow on CPU, so fewer origins
SEED = 42


def load_daily(crop):
    path, factor, name = CROPS[crop]
    df = pd.read_csv(os.path.join(BASE, path))
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["close"] / 100.0 * factor  # cents/bushel -> USD/metric ton
    df = df[["date", "price"]].sort_values("date").reset_index(drop=True)
    return df, name


def make_features(df):
    d = df.copy()
    d["lag_1"] = d["price"].shift(1)
    d["lag_2"] = d["price"].shift(2)
    d["lag_3"] = d["price"].shift(3)
    d["lag_5"] = d["price"].shift(5)
    d["lag_10"] = d["price"].shift(10)
    d["lag_20"] = d["price"].shift(20)
    d["roll_mean_5"] = d["price"].shift(1).rolling(5).mean()
    d["roll_mean_10"] = d["price"].shift(1).rolling(10).mean()
    d["roll_mean_20"] = d["price"].shift(1).rolling(20).mean()
    d["dow"] = d["date"].dt.dayofweek
    return d


FEATURES = ["lag_1", "lag_2", "lag_3", "lag_5", "lag_10", "lag_20",
            "roll_mean_5", "roll_mean_10", "roll_mean_20", "dow"]


def train_models(feat_up_to):
    """Train LR + XGBoost on feature rows up to the origin. Returns (lr, xgb)."""
    f = feat_up_to.dropna()
    X, y = f[FEATURES].values, f["price"].values
    lr = LinearRegression().fit(X, y)
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9, random_state=SEED)
    xgb.fit(X, y)
    return lr, xgb


def recursive_forecast(model, history_prices, last_date, horizon):
    """One-step model applied recursively: each prediction becomes next lag."""
    hist = list(history_prices)
    preds = []
    for step in range(horizon):
        s = pd.Series(hist)
        feats = np.array([[s.iloc[-1], s.iloc[-2], s.iloc[-3], s.iloc[-5],
                           s.iloc[-10], s.iloc[-20], s.iloc[-5:].mean(),
                           s.iloc[-10:].mean(), s.iloc[-20:].mean(),
                           (last_date + pd.offsets.BDay(step + 1)).dayofweek]])
        nxt = float(model.predict(feats)[0])
        preds.append(nxt)
        hist.append(nxt)
    return preds


def chronos_forecast(prices, horizon):
    import torch
    from chronos import ChronosPipeline
    torch.manual_seed(SEED)
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)
    ctx = torch.tensor(np.asarray(prices, dtype=np.float32))
    fc = pipeline.predict(ctx, prediction_length=horizon, num_samples=20)
    return np.median(fc.numpy(), axis=1).reshape(-1).astype(np.float64).tolist()


def mape(y_true, y_pred):
    return float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))) * 100)


def score(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse, mape(y_true, y_pred)


def run_crop(crop):
    df, name = load_daily(crop)
    feat = make_features(df)
    n = len(df)
    print(f"\n{'='*60}\n{name}: {n} trading days, "
          f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}, "
          f"last price ${df['price'].iloc[-1]:.2f}/MT")

    # ---------- backtest ----------
    origins = [n - 31 - 30 * k for k in range(N_ORIGINS)]  # each leaves >=30 actual days
    rows = []
    for oi, origin in enumerate(origins):
        origin_date = df["date"].iloc[origin]
        lr, xgb = train_models(feat.iloc[:origin + 1])
        hist = df["price"].iloc[:origin + 1].tolist()
        last_price = hist[-1]
        for h in HORIZONS:
            actual = df["price"].iloc[origin + 1:origin + 1 + h].values
            preds = {
                "Naive": [last_price] * h,
                "Linear Regression": recursive_forecast(lr, hist, origin_date, h),
                "XGBoost": recursive_forecast(xgb, hist, origin_date, h),
            }
            if oi < CHRONOS_ORIGINS:
                preds["Chronos"] = chronos_forecast(hist, h)
            for model_name, p in preds.items():
                mae, rmse, mp = score(actual, p)
                rows.append({"crop": name, "origin": str(origin_date.date()),
                             "horizon": h, "model": model_name,
                             "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                             "MAPE_%": round(mp, 2)})
        print(f"  origin {origin_date.date()}: done "
              f"(Chronos {'yes' if oi < CHRONOS_ORIGINS else 'skipped'})")
    bt = pd.DataFrame(rows)
    bt.to_csv(os.path.join(BASE, f"backtest_daily_{crop}.csv"), index=False)

    print(f"\n--- Backtest average error ({N_ORIGINS} origins, USD/MT) ---")
    summary = (bt.groupby(["horizon", "model"])[["MAE", "RMSE", "MAPE_%"]]
               .mean().round(2).reset_index())
    for h in HORIZONS:
        print(f"  {h}-day horizon:")
        for _, r in summary[summary["horizon"] == h].iterrows():
            print(f"    {r['model']:20s} MAE={r['MAE']:6.2f}  RMSE={r['RMSE']:6.2f}  MAPE={r['MAPE_%']:5.2f}%")
    best = {}
    for h in HORIZONS:
        s = summary[summary["horizon"] == h].sort_values("RMSE")
        best[h] = s.iloc[0]["model"]
    print(f"  Best by RMSE: 7-day -> {best[7]}, 30-day -> {best[30]}")

    # ---------- live forecast from the latest data ----------
    lr, xgb = train_models(feat)
    hist = df["price"].tolist()
    last_date = df["date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=30)
    live = {
        "Naive": [hist[-1]] * 30,
        "Linear Regression": recursive_forecast(lr, hist, last_date, 30),
        "XGBoost": recursive_forecast(xgb, hist, last_date, 30),
        "Chronos": chronos_forecast(hist, 30),
    }
    for h in HORIZONS:
        out = pd.DataFrame({"date": future_dates[:h].date})
        for m, p in live.items():
            out[m] = np.round(p[:h], 2)
        out.to_csv(os.path.join(BASE, f"forecast_{h}d_{crop}.csv"), index=False)
        print(f"\n  Next {h} trading days ({future_dates[0].date()} to {future_dates[h-1].date()}):")
        print(out.to_string(index=False))
    print(f"  Saved: forecast_7d_{crop}.csv, forecast_30d_{crop}.csv")

    # ---------- chart: last 120 days + 30-day forecasts ----------
    plt.figure(figsize=(12, 5))
    tail = df.iloc[-120:]
    plt.plot(tail["date"], tail["price"], label="Actual price", color="#1f77b4")
    for m, color, ls in [("Linear Regression", "#ff7f0e", "--"),
                         ("XGBoost", "#2ca02c", "--"),
                         ("Chronos", "#9467bd", "--"),
                         ("Naive", "#d62728", ":")]:
        plt.plot(future_dates, live[m], ls, label=f"{m} (30d)", color=color)
    plt.title(f"{name} price — last 120 trading days + 30-day forecast (USD/metric ton)")
    plt.xlabel("Date")
    plt.ylabel("USD / metric ton")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart = os.path.join(BASE, f"chart_daily_{crop}.png")
    plt.savefig(chart, dpi=150)
    print(f"  Chart saved: {chart}")
    return summary, best


def main():
    for crop in CROPS:
        run_crop(crop)


if __name__ == "__main__":
    main()
