"""
Crop Price Prediction API — FastAPI
====================================
Turns the trained crop-price models into a web API your own app can call.

Endpoints (all JSON):
  GET  /                              Service info (no key needed)
  GET  /health                        {"status": "ok"} (no key needed)
  GET  /models                        Models + verified backtest accuracy (no key needed)
  GET  /predict/daily?crop=wheat&days=7&model=auto
                                       Next 7 trading days (1-30). Crops: wheat, corn.
  GET  /predict/monthly?crop=wheat&months=6&model=auto
                                       Next 6 months (1-12). Crops: wheat, maize, rice.

Authentication: every /predict call needs the header
  X-API-Key: <your key>
Without it (or with a wrong key) you get 401. The key comes from the
API_KEY environment variable. If it is not set, the app uses a demo key
("demo-key-please-change") and prints a warning — set a real one in
production (see README_DEPLOY.md).

Models are trained when the server starts, from the CSV files in data/
(monthly FRED/IMF prices + daily Yahoo Finance futures). Training takes
a few seconds. Prices are USD per metric ton.

Run locally:
  API_KEY=my-secret-key uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Then open http://127.0.0.1:8000/docs for the interactive API docs.
"""

import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# ---------------------------------------------------------------- API key
API_KEY = os.environ.get("API_KEY") or "demo-key-please-change"
if API_KEY == "demo-key-please-change":
    print("WARNING: using the demo API key. Set the API_KEY environment "
          "variable to a real secret before deploying.")

# ---------------------------------------------------------------- data dir
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(BASE, "..", "data")

# ------------------------------------------------- verified backtest (2026-11-13)
# Daily: average error over 6 backtest origins. Monthly: last-12-months test.
BACKTEST_DAILY = {
    ("wheat", 7): {
        "naive": {"mae": 9.41, "rmse": 10.99, "mape_pct": 3.90},
        "linear_regression": {"mae": 9.43, "rmse": 11.05, "mape_pct": 3.91},
        "xgboost": {"mae": 11.01, "rmse": 12.54, "mape_pct": 4.57},
        "chronos": {"mae": 12.81, "rmse": 15.37, "mape_pct": 4.97},
    },
    ("wheat", 30): {
        "naive": {"mae": 14.20, "rmse": 16.20, "mape_pct": 6.06},
        "linear_regression": {"mae": 14.44, "rmse": 16.36, "mape_pct": 6.15},
        "xgboost": {"mae": 17.15, "rmse": 18.84, "mape_pct": 7.26},
        "chronos": {"mae": 18.44, "rmse": 21.10, "mape_pct": 7.63},
    },
    ("corn", 7): {
        "naive": {"mae": 5.14, "rmse": 5.78, "mape_pct": 2.80},
        "linear_regression": {"mae": 5.43, "rmse": 6.10, "mape_pct": 2.96},
        "xgboost": {"mae": 5.53, "rmse": 6.22, "mape_pct": 2.98},
        "chronos": {"mae": 6.71, "rmse": 7.51, "mape_pct": 3.51},
    },
    ("corn", 30): {
        "naive": {"mae": 10.64, "rmse": 11.75, "mape_pct": 5.86},
        "linear_regression": {"mae": 10.47, "rmse": 11.53, "mape_pct": 5.78},
        "xgboost": {"mae": 11.54, "rmse": 12.79, "mape_pct": 6.34},
        "chronos": {"mae": 12.13, "rmse": 13.30, "mape_pct": 6.37},
    },
}
BACKTEST_MONTHLY = {
    "wheat": {
        "naive": {"mae": 36.65, "rmse": 44.57, "mape_pct": 17.66},
        "linear_regression": {"mae": 11.45, "rmse": 15.11, "mape_pct": 5.65},
        "xgboost": {"mae": 11.79, "rmse": 14.81, "mape_pct": 5.68},
        "chronos": {"mae": 36.21, "rmse": 44.16, "mape_pct": 16.87},
    },
    "maize": {
        "naive": {"mae": 12.47, "rmse": 18.27, "mape_pct": 5.68},
        "linear_regression": {"mae": 7.36, "rmse": 10.49, "mape_pct": 3.48},
        "xgboost": {"mae": 9.89, "rmse": 13.44, "mape_pct": 4.57},
    },
    "rice": {
        "naive": {"mae": 78.09, "rmse": 91.93, "mape_pct": 20.24},
        "linear_regression": {"mae": 16.52, "rmse": 22.36, "mape_pct": 4.17},
        "xgboost": {"mae": 23.66, "rmse": 25.02, "mape_pct": 5.95},
    },
}
# Default ("auto") model per crop: the verified winner among trained models.
AUTO_MONTHLY = {"wheat": "xgboost", "maize": "linear_regression", "rice": "linear_regression"}
AUTO_DAILY = {"wheat": "linear_regression", "corn": "linear_regression"}

MONTHLY_FILES = {
    "wheat": ("PWHEAMTUSDM.csv", "PWHEAMTUSDM", 1.0),
    "maize": ("PMAIZMTUSDM.csv", "PMAIZMTUSDM", 1.0),
    "rice": ("PRICENPQUSDM.csv", "PRICENPQUSDM", 1.0),
}
DAILY_FILES = {
    # cents/bushel -> USD/metric ton
    "wheat": ("daily_zw.csv", 36.7437),
    "corn": ("daily_zc.csv", 39.3680),
}

MONTHLY_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
                    "roll_mean_3", "roll_mean_6", "roll_mean_12", "month"]
DAILY_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_5", "lag_10", "lag_20",
                  "roll_mean_5", "roll_mean_10", "roll_mean_20", "dow"]


def _monthly_features(df):
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
    return d


def _daily_features(df):
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


def _train(df, features):
    f = df.dropna()
    X, y = f[features].values, f["price"].values
    lr = LinearRegression().fit(X, y)
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb.fit(X, y)
    return {"linear_regression": lr, "xgboost": xgb}


def _load_monthly(crop):
    fname, col, _ = MONTHLY_FILES[crop]
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["date"] = pd.to_datetime(df["observation_date"])
    df["price"] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)


def _load_daily(crop):
    fname, factor = DAILY_FILES[crop]
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["close"] / 100.0 * factor
    return df[["date", "price"]].sort_values("date").reset_index(drop=True)


def _recursive_forecast_monthly(model, history, last_date, months):
    hist = list(history)
    out = []
    for step in range(months):
        s = pd.Series(hist)
        nxt_date = last_date + pd.offsets.MonthBegin(step + 1)
        feats = np.array([[s.iloc[-1], s.iloc[-2], s.iloc[-3], s.iloc[-6],
                           s.iloc[-12], s.iloc[-3:].mean(), s.iloc[-6:].mean(),
                           s.iloc[-12:].mean(), nxt_date.month]])
        nxt = float(model.predict(feats)[0])
        out.append((nxt_date.date(), round(nxt, 2)))
        hist.append(nxt)
    return out


def _recursive_forecast_daily(model, history, last_date, days):
    hist = list(history)
    out = []
    for step in range(days):
        s = pd.Series(hist)
        nxt_date = last_date + pd.offsets.BDay(step + 1)
        feats = np.array([[s.iloc[-1], s.iloc[-2], s.iloc[-3], s.iloc[-5],
                           s.iloc[-10], s.iloc[-20], s.iloc[-5:].mean(),
                           s.iloc[-10:].mean(), s.iloc[-20:].mean(),
                           nxt_date.dayofweek]])
        nxt = float(model.predict(feats)[0])
        out.append((nxt_date.date(), round(nxt, 2)))
        hist.append(nxt)
    return out


# ------------------------------------------------------- train on startup
print("Training models (this takes a few seconds)...")
MONTHLY = {}
for crop in MONTHLY_FILES:
    df = _load_monthly(crop)
    MONTHLY[crop] = {"df": df, "models": _train(_monthly_features(df), MONTHLY_FEATURES)}
    print(f"  monthly {crop}: {len(df)} months to {df['date'].iloc[-1].date()}")
DAILY = {}
for crop in DAILY_FILES:
    df = _load_daily(crop)
    DAILY[crop] = {"df": df, "models": _train(_daily_features(df), DAILY_FEATURES)}
    print(f"  daily {crop}: {len(df)} trading days to {df['date'].iloc[-1].date()}")
print("Models ready.")

# ------------------------------------------------------------------- app
app = FastAPI(title="Crop Price Prediction API",
              description="Wheat, maize/corn and rice price forecasts. "
                          "Send your API key in the X-API-Key header.",
              version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # your own web app can call this API from the browser
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _check_key(x_api_key: str = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401,
                            detail="Missing or wrong API key. Send it in the X-API-Key header.")
    return True


@app.get("/")
def root():
    return {
        "service": "Crop Price Prediction API",
        "version": "1.0",
        "crops_daily": sorted(DAILY_FILES),
        "crops_monthly": sorted(MONTHLY_FILES),
        "endpoints": [
            "GET /health",
            "GET /models",
            "GET /predict/daily?crop=wheat&days=7&model=auto (needs X-API-Key)",
            "GET /predict/monthly?crop=wheat&months=6&model=auto (needs X-API-Key)",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    daily = {f"{crop}_{h}d": table for (crop, h), table in BACKTEST_DAILY.items()}
    return {
        "daily_backtest": daily,
        "monthly_backtest": BACKTEST_MONTHLY,
        "note": "Daily backtest = average error over 6 past origins. "
                "Monthly backtest = last-12-months test. Lower RMSE is better.",
    }


@app.get("/predict/daily")
def predict_daily(crop: str = Query("wheat"),
                  days: int = Query(7, ge=1, le=30),
                  model: str = Query("auto"),
                  x_api_key: str = Header(default=None)):
    _check_key(x_api_key)
    crop = crop.lower()
    if crop not in DAILY:
        raise HTTPException(status_code=400,
                            detail=f"Unknown crop '{crop}'. Use: {sorted(DAILY)}")
    model = model.lower()
    if model == "auto":
        model = AUTO_DAILY[crop]
    if model == "naive":
        df = DAILY[crop]["df"]
        last_price = round(float(df["price"].iloc[-1]), 2)
        dates = pd.bdate_range(df["date"].iloc[-1] + pd.offsets.BDay(1), periods=days)
        forecast = [{"date": str(d.date()), "price": last_price} for d in dates]
    elif model in ("linear_regression", "xgboost"):
        df = DAILY[crop]["df"]
        preds = _recursive_forecast_daily(DAILY[crop]["models"][model],
                                          df["price"].tolist(), df["date"].iloc[-1], days)
        forecast = [{"date": str(d), "price": p} for d, p in preds]
    else:
        raise HTTPException(status_code=400,
                            detail="model must be auto, naive, linear_regression or xgboost")
    df = DAILY[crop]["df"]
    bucket = 7 if days <= 7 else 30
    return {
        "crop": crop, "type": "daily", "horizon_days": days, "model": model,
        "unit": "USD per metric ton",
        "last_price": round(float(df["price"].iloc[-1]), 2),
        "last_date": str(df["date"].iloc[-1].date()),
        "backtest": BACKTEST_DAILY[(crop, bucket)][model],
        "backtest_note": f"Average error over 6 backtest origins ({bucket}-day horizon).",
        "forecast": forecast,
    }


@app.get("/predict/monthly")
def predict_monthly(crop: str = Query("wheat"),
                    months: int = Query(6, ge=1, le=12),
                    model: str = Query("auto"),
                    x_api_key: str = Header(default=None)):
    _check_key(x_api_key)
    crop = crop.lower()
    if crop not in MONTHLY:
        raise HTTPException(status_code=400,
                            detail=f"Unknown crop '{crop}'. Use: {sorted(MONTHLY)}")
    model = model.lower()
    if model == "auto":
        model = AUTO_MONTHLY[crop]
    df = MONTHLY[crop]["df"]
    if model == "naive":
        last_price = round(float(df["price"].iloc[-1]), 2)
        dates = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1),
                              periods=months, freq="MS")
        forecast = [{"date": str(d.date()), "price": last_price} for d in dates]
    elif model in ("linear_regression", "xgboost"):
        preds = _recursive_forecast_monthly(MONTHLY[crop]["models"][model],
                                            df["price"].tolist(), df["date"].iloc[-1], months)
        forecast = [{"date": str(d), "price": p} for d, p in preds]
    else:
        raise HTTPException(status_code=400,
                            detail="model must be auto, naive, linear_regression or xgboost")
    return {
        "crop": crop, "type": "monthly", "horizon_months": months, "model": model,
        "unit": "USD per metric ton",
        "last_price": round(float(df["price"].iloc[-1]), 2),
        "last_date": str(df["date"].iloc[-1].date()),
        "backtest": BACKTEST_MONTHLY[crop][model],
        "backtest_note": "Error on the last-12-months test set.",
        "forecast": forecast,
    }
