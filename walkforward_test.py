"""Walk-forward test — is the model REALLY good, or just lucky on one window?

One 12-month test (the original check) can flatter a model. This script
re-runs the exact same pipeline on SIX different 12-month test windows
(ending Sep 2021 .. Sep 2026), so we see whether the accuracy holds up
across calm years AND the 2022 Ukraine-war price shock.

Same models, same features, same code as crop_price_prediction.py:
  Seasonal Naive vs Linear Regression vs XGBoost.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "wheat": ("data/PWHEAMTUSDM.csv", "PWHEAMTUSDM", "Wheat"),
    "maize": ("data/PMAIZMTUSDM.csv", "PMAIZMTUSDM", "Maize (corn)"),
    "rice": ("data/PRICENPQUSDM.csv", "PRICENPQUSDM", "Rice"),
}
TEST_MONTHS = 12
# Test windows end in Sep of each year (data ends Sep 2026).
ORIGIN_ENDS = ["2021-09-01", "2022-09-01", "2023-09-01",
               "2024-09-01", "2025-09-01", "2026-09-01"]


def load_prices(crop):
    path, col, name = FILES[crop]
    df = pd.read_csv(os.path.join(BASE, path))
    df["date"] = pd.to_datetime(df["observation_date"])
    df["price"] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True), name


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


def run_origin(df, test_end):
    """One 12-month test window ending at test_end. Returns {model: (mae, rmse, mape)}."""
    test_end = pd.Timestamp(test_end)
    test_start = test_end - pd.offsets.MonthBegin(TEST_MONTHS) + pd.offsets.MonthBegin(1)
    sub = df[df["date"] <= test_end].copy()
    feat = make_features(sub)
    cols = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
            "roll_mean_3", "roll_mean_6", "roll_mean_12", "month"]
    is_test = feat["date"] >= test_start
    X_train, X_test = feat.loc[~is_test, cols].values, feat.loc[is_test, cols].values
    y_train, y_test = feat.loc[~is_test, "price"].values, feat.loc[is_test, "price"].values
    if len(y_test) != TEST_MONTHS:
        return None
    preds = {
        "Seasonal Naive": feat.loc[is_test, "lag_12"].values,
        "Linear Regression": LinearRegression().fit(X_train, y_train).predict(X_test),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.9, colsample_bytree=0.9,
                                random_state=42).fit(X_train, y_train).predict(X_test),
    }
    out = {}
    for name, pred in preds.items():
        out[name] = (mean_absolute_error(y_test, pred),
                     float(np.sqrt(mean_squared_error(y_test, pred))),
                     mape(y_test, pred))
    return out


def main():
    all_rows = []
    for crop in ["wheat", "maize", "rice"]:
        df, name = load_prices(crop)
        print(f"\n{'=' * 64}\n{name} — walk-forward, 6 x 12-month test windows")
        for end in ORIGIN_ENDS:
            res = run_origin(df, end)
            if res is None:
                print(f"  window ending {end}: not enough data, skipped")
                continue
            label = f"{pd.Timestamp(end) - pd.offsets.MonthBegin(12) + pd.offsets.MonthBegin(1):%Y-%m} to {end[:7]}"
            line = f"  {label}: "
            for model in ["Seasonal Naive", "Linear Regression", "XGBoost"]:
                mae, rmse, mp = res[model]
                line += f"{model} MAPE {mp:5.2f}%  "
                all_rows.append({"crop": name, "window_end": end, "model": model,
                                 "MAE": mae, "RMSE": rmse, "MAPE": mp})
            print(line)
        sub = pd.DataFrame([r for r in all_rows if r["crop"] == name])
        print(f"\n  --- {name}: average across all 6 windows ---")
        summ = sub.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().round(2)
        print(summ.to_string())
        wins = sub.loc[sub.groupby("window_end")["RMSE"].idxmin(), "model"].value_counts()
        print("  Wins by lowest RMSE:", dict(wins))
        worst = sub.loc[sub.groupby("model")["MAPE"].idxmax()]
        print("  Worst single window MAPE per model:")
        for _, r in worst.iterrows():
            print(f"    {r['model']:20s} {r['MAPE']:5.2f}% (window ending {r['window_end'][:7]})")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(BASE, "walkforward_results.csv"), index=False)
    print("\nFull results saved: walkforward_results.csv")


if __name__ == "__main__":
    main()
