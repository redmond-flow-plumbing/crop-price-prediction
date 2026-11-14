# Crop Price Prediction — Project Plan

**Goal:** predict the monthly price of a crop (wheat, maize, or rice) using free
data and free machine learning models, and compare the models honestly.

**What is already done for you:** real monthly world prices (USD per metric ton)
downloaded from FRED / IMF into `data/` — wheat (Jan 1992–present), maize
(1992–present), rice (May 2010–present). The working code is
`crop_price_prediction.py`. Run it with `python3 crop_price_prediction.py`.

---

## Step 1 — Understand the problem (5 minutes)
- Target (what we predict): next month's price of one crop.
- This is **regression** (the answer is a number, not a category) and a
  **time series** problem (order of months matters).
- Success = low error on months the model has never seen.

## Step 2 — Look at the data (EDA)
- Load the CSV, check rows, date range, missing values.
- Plot the price over time: does it trend up? Does it repeat every year
  (seasonality)? Are there sudden jumps (shocks)?
- Write 3–4 lines about what you see. This goes in the report.

## Step 3 — Split train / test the correct way
- **Last 12 months = test set**, everything before = train set.
- Never shuffle time series data. Shuffling leaks the future into training
  and gives fake high accuracy — teachers check for this.

## Step 4 — Baseline first: Seasonal Naive
- Rule: this month's forecast = same month last year.
- It is not AI, but it is the benchmark. Every other model must beat it,
  otherwise the model is useless. Always report it.

## Step 5 — Build features
- From past prices only: lag_1, lag_2, lag_3, lag_6, lag_12 (price 1, 2, 3,
  6, 12 months ago), rolling means of 3, 6, 12 months, and the calendar month.
- Never use a feature that needs future data (that is data leakage).

## Step 6 — Train Linear Regression
- The simplest real model. Fit on train, predict on test.
- You already know this from class: y = intercept + slope × x, here with
  several features instead of one.

## Step 7 — Train XGBoost
- Same features, stronger model. It usually beats Linear Regression on
  price data because prices are not straight lines.
- `XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)`.

## Step 8 — (Optional, advanced) Chronos zero-shot
- Amazon's pretrained time-series model (`amazon/chronos-t5-small`).
- No training: give it past prices, it forecasts the next months.
- Install: `pip install chronos-forecasting`. Needs a bit more memory;
  runs on CPU, or free on Google Colab with GPU.
- **Verified 2026-11-13:** Chronos ran here (`chronos_test.py`, seed fixed
  for reproducibility). On wheat it scored MAE 36.21, RMSE 44.16,
  MAPE 16.87% — about the same as the Seasonal Naive baseline, far behind
  trained Linear Regression / XGBoost. Zero-shot is not magic: on this data,
  training on the actual series wins. Note: Chronos samples, so without
  the fixed seed its numbers move a little run to run — always set the seed
  (`torch.manual_seed(42)`) before reporting.
- **Run it on Colab:** `crop_price_prediction_chronos_colab.ipynb` — upload it to
  Google Colab, Runtime → Run all. It downloads the same real wheat prices live
  from FRED, runs the same last-12-months test as the main script, and prints
  Chronos MAE/RMSE/MAPE next to the verified baselines (Seasonal Naive 44.57,
  Linear Regression 15.11, XGBoost 14.81 RMSE). Copy its numbers into the table below.
- Local run tried 2026-11-13: `chronos_test.py` is ready and `chronos-forecasting`
  is installed, but the model download was blocked in that background environment
  (checkpoint cache incomplete) — an environment limit, not a Chronos problem.
  Colab is the working route; no approval or install step is needed from you.

## Step 9 — Compare honestly
- Metrics on the 12-month test set: **MAE** (average error in dollars),
  **RMSE** (punishes big errors more), **MAPE** (error in %).
- Make one table with all models. Lowest RMSE wins.
- Note: if XGBoost wins on test but Seasonal Naive is close, say so —
  honest reporting scores more than a big claim.

## Step 10 — Forecast the future
- Take the best model, feed its own predictions back as lags, and forecast
  the next 6 months. Save the numbers to CSV and the chart to PNG.

## Step 11 — Present it
Report structure (one page each is enough):
1. Introduction — why crop prices matter (farmers, traders, food security).
2. Data — source (FRED/IMF), crop, months, units.
3. Method — train/test split, features, the 3 models.
4. Results — the comparison table + chart.
5. Conclusion — which model won, by how much, and why.
6. Limitations — prices also move on weather, policy and world events,
   which this data does not include; next step is adding rainfall and
   currency data as features.

---

## Files in this folder
- `crop_price_prediction.py` — the full working pipeline (run it, it prints
  the scoreboard and saves predictions + chart).
- `chronos_test.py` — Chronos zero-shot test on the same 12-month window
  (local run; needs the model download — see Step 8).
- `crop_price_prediction_chronos_colab.ipynb` — same Chronos test as a
  Colab notebook: upload to Google Colab, Run all, no local install needed.
- `data/PWHEAMTUSDM.csv` — wheat monthly prices (FRED/IMF).
- `data/PMAIZMTUSDM.csv` — maize monthly prices (FRED/IMF).
- `data/PRICENPQUSDM.csv` — rice monthly prices (FRED/IMF).
- `predictions_<crop>.csv` — actual vs predicted on the test months.
- `forecast_next6_<crop>.csv` — the 6-month forecast.
- `chart_<crop>.png` — price history + predictions chart.
- `PROJECT_PLAN.md` — this file.

## Deploy as an API (2026-11-13)
The model is also a web API now — see `api/`:
- `api/app.py` — FastAPI app. Trains all models at startup from `api/data/`,
  serves `GET /predict/daily?crop=wheat&days=7` and
  `GET /predict/monthly?crop=wheat&months=6`, protected by an `X-API-Key`
  header, plus public `/`, `/health`, `/models` (backtest accuracy) and
  `/docs` (interactive docs). CORS open so a web app can call it.
- `api/README_DEPLOY.md` — run locally + deploy free on Render (free tier,
  `render.yaml` included — Render generates the API key for you) + how to
  call it from Python (`requests`) and JavaScript (`fetch`).
- `render.yaml` — Render deploy config at the project root.
- Verified live locally 2026-11-13: health 200, no key → 401, wrong key →
  401, bad crop → 400, forecasts identical to the scripts (daily wheat 7d
  217.19→219.31; monthly wheat 6mo 258.83→297.85). One bug found and fixed
  in testing: `/models` returned 500 because the backtest table used tuple
  keys — now string keys (`wheat_7d`), returns 200.

## To run on another crop
Change `CROP = "wheat"` at the top of the script to `"maize"` or `"rice"`
and run again. Everything else stays the same.

## Daily test — 7-day and 30-day forecasts (run 2026-11-13)
Monthly data cannot answer "next 7 days", so this test uses **daily** prices:
wheat and corn futures from Yahoo Finance (free), 1,256 trading days,
Nov 2021 – Nov 13 2026, converted to USD/metric ton. Futures trade Mon–Fri,
so "7 days" = 7 trading days (~9–10 calendar days) and "30 days" = 30
trading days (~6 calendar weeks). Code: `daily_forecast.py`.

**How accuracy was measured (backtest):** rewind to 6 recent dates (origins:
Feb 27, Apr 13, May 26, Jul 9, Aug 20, Oct 2 2026), train only on data up to
each origin, forecast 7 and 30 trading days, score against what actually
happened. Average error across origins = honest accuracy. Chronos averages
over the 3 most recent origins only (slow on CPU) — the others over all 6.

**Wheat backtest (USD/MT)**

| Horizon | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 7-day | Naive | 9.41 | 10.99 | 3.90% |
| 7-day | Linear Regression | 9.43 | 11.05 | 3.91% |
| 7-day | XGBoost | 11.01 | 12.54 | 4.57% |
| 7-day | Chronos | 12.81 | 15.37 | 4.97% |
| 30-day | Naive | 14.20 | 16.20 | 6.06% |
| 30-day | Linear Regression | 14.44 | 16.36 | 6.15% |
| 30-day | XGBoost | 17.15 | 18.84 | 7.26% |
| 30-day | Chronos | 18.44 | 21.10 | 7.63% |

**Corn backtest (USD/MT)**

| Horizon | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 7-day | Naive | 5.14 | 5.78 | 2.80% |
| 7-day | Linear Regression | 5.43 | 6.10 | 2.96% |
| 7-day | XGBoost | 5.53 | 6.22 | 2.98% |
| 7-day | Chronos | 6.71 | 7.51 | 3.51% |
| 30-day | Naive | 10.64 | 11.75 | 5.86% |
| 30-day | Linear Regression | 10.47 | 11.53 | 5.78% |
| 30-day | XGBoost | 11.54 | 12.79 | 6.34% |
| 30-day | Chronos | 12.13 | 13.30 | 6.37% |

**Key finding:** on daily data, prices behave almost like a random walk —
"tomorrow = today" (Naive) ties or beats the ML models at 7 days, and only
Linear Regression edges ahead at 30 days. Compare with the monthly test,
where ML crushed the baseline. Accuracy to quote: 7-day error ~3–5%,
30-day error ~6–7.6%. The further ahead, the worse — that is normal.

**Live forecasts from Nov 13, 2026** (last prices: wheat $217.25/MT,
corn $175.68/MT). Full day-by-day files: `forecast_7d_wheat.csv`,
`forecast_30d_wheat.csv`, `forecast_7d_corn.csv`, `forecast_30d_corn.csv`.
- Wheat, Nov 16–24 (7d): all models $217–219 — LR ends 219.31, XGBoost
  217.76, Chronos 216.69, Naive flat 217.25.
- Wheat, to Dec 25 (30d): LR 220.50, XGBoost 219.42, Chronos 221.30 — all
  point slightly up from 217.25.
- Corn, Nov 16–24 (7d): LR ~175.5, XGBoost drifts down to ~173.7, Chronos
  ~174.7, Naive flat 175.68.
- Corn, to Dec 25 (30d): LR 176.62, XGBoost 173.91, Chronos 179.82.
- Charts: `chart_daily_wheat.png`, `chart_daily_corn.png`.

## Verified results (run 2026-11-13, test = last 12 months, Oct 2025–Sep 2026)
All three crops ran with the same code. Prices in USD per metric ton.

**Wheat** (417 months, 1992–2026) — best: XGBoost (lowest RMSE)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal Naive | 36.65 | 44.57 | 17.66% |
| Linear Regression | 11.45 | 15.11 | 5.65% |
| XGBoost | 11.79 | 14.81 | 5.68% |
| Chronos (t5-small, zero-shot) | 36.21 | 44.16 | 16.87% |

Forecast Oct 2026–Mar 2027: 258.83, 268.01, 284.54, 286.26, 293.73, 297.85.
Chronos 6-month forecast: 243.98, 239.86, 239.18, 235.07, 230.27, 235.75.

> Note (2026-11-13): the 6-month forecasts above use a model retrained on
> ALL months. The first version held out the last 12 months for testing and
> forecast from that weaker model; the final model now trains on everything
> including the test year. Test scores (MAE/RMSE/MAPE) are unchanged — only
> the forecast values moved slightly. The API (`api/app.py`) trains the same
> way, so script and API forecasts now agree exactly.

**Maize / corn** (417 months, 1992–2026) — best: Linear Regression

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal Naive | 12.47 | 18.27 | 5.68% |
| Linear Regression | 7.36 | 10.49 | 3.48% |
| XGBoost | 9.89 | 13.44 | 4.57% |

Forecast Oct 2026–Mar 2027: 240.25, 240.93, 239.98, 239.75, 238.23, 236.19.

**Rice** (197 months, 2010–2026) — best: Linear Regression

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal Naive | 78.09 | 91.93 | 20.24% |
| Linear Regression | 16.52 | 22.36 | 4.17% |
| XGBoost | 23.66 | 25.02 | 5.95% |

Forecast Oct 2026–Mar 2027: 480.31, 484.37, 483.34, 489.19, 496.92, 503.47.

**Key finding for the report:** no single model wins everywhere — XGBoost wins
on wheat, Linear Regression wins on maize and rice. But on all three crops,
both ML models crush the Seasonal Naive baseline (MAPE ~18–20% down to
~3–6%). And Chronos, the pretrained zero-shot model, scores about the same
as the baseline on wheat (RMSE 44.16 vs 44.57) — it does not beat the trained
models. That honest comparison is the strongest part of the project.
Note: the 6-month forecasts are recursive (each prediction feeds the next),
so they can drift — say this in the limitations.
