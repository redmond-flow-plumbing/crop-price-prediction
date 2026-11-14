# Deploy the Crop Price API + Use It In Your App

Your trained model is now a web API: any app can ask it for a forecast over
the internet, using an **API key** (a secret password for the API).

## How it works

```
Your app  --(X-API-Key header)-->  https://your-api.onrender.com/predict/daily?crop=wheat&days=7
                                   <-- {"forecast": [{"date": "2026-11-16", "price": 217.19}, ...]}
```

## 1. Run it on your own computer first (test)

In the `api/` folder:

```
pip install -r requirements.txt
API_KEY=my-secret-key uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On Windows PowerShell the first line of the second command is:
`$env:API_KEY="my-secret-key"` and then `uvicorn app:app --port 8000 --reload`.

Then try it (new terminal):

```
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/predict/daily?crop=wheat&days=7
curl -H "X-API-Key: my-secret-key" "http://127.0.0.1:8000/predict/daily?crop=wheat&days=7"
```

The first predict call without a key returns **401** (that is the point — the
key protects the API). The one with the key returns the forecast. Open
http://127.0.0.1:8000/docs in a browser for interactive API docs.

## 2. Deploy it free on Render (so your app can reach it from anywhere)

Render's free tier is enough for a student project (the server sleeps after
15 minutes without traffic, so the first call after a pause takes ~30–60
seconds to wake up — that is normal on the free tier).

1. Create a free account on **github.com** and **render.com**.
2. Put this whole `crop-price-prediction` folder in a new GitHub repo
   (GitHub Desktop is the easiest way: create repo, drag the folder in,
   commit, publish).
3. In Render: **New + → Web Service → pick your repo**.
4. Render finds `render.yaml` in the folder and fills everything in
   (Python, `pip install -r requirements.txt`, `uvicorn app:app`).
   Click **Create Web Service**. The first deploy takes 2–4 minutes.
5. When it is live you get a URL like
   `https://crop-price-api-xxxx.onrender.com`.
6. **Get your API key:** in Render, open the service → **Environment** →
   the `API_KEY` variable was generated for you — click the eye icon to
   reveal it and copy it. That is the key your app must send. Never post
   it publicly; if it leaks, generate a new value there.

Other free options if Render does not suit you: **Railway** (free trial
credits, then pay-as-you-go), **Hugging Face Spaces** (free, Docker SDK),
**PythonAnywhere** (free tier, needs a WSGI adapter for FastAPI).
Render is the simplest for this app.

## 3. Use the API key in your own application

Every `/predict` call must send the header `X-API-Key: <your key>`.
The `/`, `/health` and `/models` endpoints need no key.

**Python (requests):** — or just run the ready-made client in this folder:
`API_URL=https://crop-price-api-xxxx.onrender.com API_KEY=<your key> python example_client.py`
(it prints the 7-day + 6-month wheat forecasts in USD/ton and PKR/maund, and
shows a clear error if the key is wrong). The snippet below does the same thing:

```python
import requests

API_URL = "https://crop-price-api-xxxx.onrender.com"
API_KEY = "paste-your-key-here"   # keep it in an env var, not in shared code

r = requests.get(
    f"{API_URL}/predict/daily",
    params={"crop": "wheat", "days": 7, "model": "auto"},
    headers={"X-API-Key": API_KEY},
    timeout=60,
)
r.raise_for_status()
data = r.json()
print("Last price:", data["last_price"], data["unit"])
for row in data["forecast"]:
    print(row["date"], row["price"])
print("Backtest accuracy (MAPE %):", data["backtest"]["mape_pct"])
```

**JavaScript (fetch, for a website):**

```javascript
const API_URL = "https://crop-price-api-xxxx.onrender.com";
const API_KEY = "paste-your-key-here"; // for a real site, call through your own backend so the key is not visible in the browser

const res = await fetch(`${API_URL}/predict/daily?crop=wheat&days=7&model=auto`, {
  headers: { "X-API-Key": API_KEY }
});
const data = await res.json();
console.log(data.forecast);
```

**Endpoints:**

| Endpoint | What it returns |
|---|---|
| `GET /predict/daily?crop=wheat&days=7` | Next 7 trading days. `crop`: wheat, corn. `days`: 1–30. `model`: auto, naive, linear_regression, xgboost. |
| `GET /predict/monthly?crop=wheat&months=6` | Next 6 months. `crop`: wheat, maize, rice. `months`: 1–12. `model`: auto, naive, linear_regression, xgboost. |
| `GET /models` | All models + verified backtest accuracy (no key needed). |

Every forecast response also includes `backtest` (MAE, RMSE, MAPE %) so your
app can show the accuracy next to the prediction. Prices are USD per metric
ton; the daily last price for wheat is the futures price converted from
cents/bushel.

## 4. Changing the key

If the key leaks, change it: in Render → Environment → edit `API_KEY` →
save → the service restarts with the new key. Locally, just restart the
server with a new `API_KEY` value. To make a strong key yourself:

```
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 5. Keeping the data fresh

The API trains from the CSV files in `api/data/` when it starts. To update
the prices, replace those CSVs with fresh downloads (same filenames and
columns) and redeploy/restart — the models retrain automatically. Monthly:
FRED `fredgraph.csv?id=PWHEAMTUSDM` (wheat), `PMAIZMTUSDM` (maize),
`PRICENPQUSDM` (rice). Daily: Yahoo Finance `ZW=F` (wheat), `ZC=F` (corn).
