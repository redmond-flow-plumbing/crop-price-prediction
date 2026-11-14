# Crop Price Prediction — Run It On Your Own Computer

A complete, working student project: predict crop prices (wheat, maize/corn,
rice) with free data and free machine-learning models, and compare them
honestly. Everything was run and verified on 2026-11-13 — see
`PROJECT_PLAN.md` for the full results.

## What is inside

| File | What it does |
|---|---|
| `crop_price_prediction.py` | Monthly prices: Seasonal Naive vs Linear Regression vs XGBoost, last-12-months test, 6-month forecast + chart. Change `CROP = "wheat"` to `"maize"` or `"rice"`. |
| `daily_forecast.py` | Daily prices: 7-day + 30-day forecasts with a real backtest (6 past dates, scored against what actually happened) + live forecasts + charts. Wheat + corn. |
| `chronos_test.py` | Chronos (Amazon's pre-trained time-series model, zero-shot) on the same monthly wheat test. Needs torch (see setup). |
| `crop_price_prediction_chronos_colab.ipynb` | Same Chronos test as a Google Colab notebook (free GPU, no install). |
| `data/` | All training data, already downloaded — no internet needed to run the two main scripts: monthly FRED/IMF prices (wheat + maize 1992–Sep 2026, rice 2010–Sep 2026) and daily Yahoo Finance futures (wheat + corn, Nov 2021–Nov 13 2026). |
| `PROJECT_PLAN.md` | 11-step project plan + all verified results and the report structure. |
| `predictions_*.csv`, `forecast_*.csv`, `backtest_*.csv` | Results from the verified runs. |
| `chart_*.png` | Charts from the verified runs. |
| `api/` | The model as a web API (FastAPI): `app.py`, `README_DEPLOY.md` (Render deploy + how to use the API key in your own app), `render.yaml`, and `example_client.py` — a ready-to-run client that calls the API and prints wheat forecasts in USD/ton and PKR/maund. Run it with `API_URL=... API_KEY=... python example_client.py`. |

## Setup (Windows, Mac, or Linux)

1. Install **Python 3.10 or newer** from python.org. During install on
   Windows, tick **"Add Python to PATH"**.
2. Open a terminal (Windows: Command Prompt or PowerShell) in this folder and run:

```
pip install -r requirements.txt
```

That installs pandas, numpy, scikit-learn, xgboost, matplotlib, yfinance.
The monthly and daily scripts run with only these.

3. **Only for Chronos** (`chronos_test.py`): install the CPU version of
   PyTorch first (small, no GPU needed), then Chronos:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chronos-forecasting
```

The first Chronos run downloads the model (~200 MB) from Hugging Face.
It needs internet for that one download; after that it works offline.

## How to run

```
python crop_price_prediction.py   # monthly: wheat (or change CROP), prints the scoreboard
python daily_forecast.py          # daily: 7-day + 30-day backtest and live forecasts (takes a few minutes)
python chronos_test.py            # Chronos zero-shot on monthly wheat (needs step 3)
```

Each script prints its results and saves predictions, forecasts and charts
in this same folder. If you change `CROP` in `crop_price_prediction.py`
to `"maize"` or `"rice"`, run it again for that crop.

## Getting fresh data (optional)

The `data/` files are already included, so the scripts run as-is. To refresh
them later: monthly prices come from
https://fred.stlouisfed.org/graph/fredgraph.csv?id=PWHEAMTUSDM (wheat),
`PMAIZMTUSDM` (maize), `PRICENPQUSDM` (rice). Daily futures can be
re-downloaded with yfinance: `yf.download("ZW=F", period="5y", interval="1d")`
(wheat) and `"ZC=F"` (corn), then convert cents/bushel to USD/metric ton
(wheat x36.7437, corn x39.3680) — see `daily_forecast.py`.

## Verified results (2026-11-13)

- **Monthly, wheat:** XGBoost best (RMSE 14.81, MAPE 5.68%); maize and rice:
  Linear Regression best (RMSE 10.49 / 22.36). Baseline Seasonal Naive MAPE
  17.66% / 5.68% / 20.24% — ML wins clearly on monthly data.
- **Monthly, Chronos (zero-shot):** RMSE 44.16, MAPE 16.87% — about the same
  as the baseline, far behind trained models.
- **Daily, wheat 7-day:** Naive best (RMSE 10.99, MAPE 3.90%); 30-day: Naive
  best (RMSE 16.20, MAPE 6.06%). **Daily, corn 7-day:** Naive best (5.78,
  2.80%); 30-day: Linear Regression best (11.53, 5.78%). Daily prices are
  close to a random walk, so "tomorrow = today" is hard to beat at 7 days.

## Verification (2026-11-13) — checked so nothing errors on your computer

- **Data (all 5 files):** 0 missing values, 0 duplicate dates, 0 non-numeric
  values, 0 zeros/negatives, dates in order. Daily gaps max 4 days
  (weekend/holiday — normal). The biggest price jumps are real events, not
  data errors: Jul 2012 US drought, Mar 2022 Ukraine war, Aug 2023 India
  rice export ban.
- **Code (clean-copy test):** the zip was extracted to a fresh folder and
  all three scripts were run from there — `crop_price_prediction.py`,
  `daily_forecast.py`, `chronos_test.py` all finished with exit code 0,
  0 tracebacks, and reproduced the exact results above. The Colab notebook
  was fixed to the current Chronos API (positional `predict` argument,
  median over samples, fixed seed) and its JSON was validated.
