"""
Example client — call your Crop Price API from your own app
=============================================================
This is the ready-to-run version of the Python snippet in
README_DEPLOY.md. It shows exactly how your application uses the API
key: the key goes in the ``X-API-Key`` header of every /predict call.

Setup (no server needed yet — it works against the local test server):
  1. Start the API in the api/ folder:
         API_KEY=my-secret-key uvicorn app:app --port 8000
     (On Windows PowerShell: $env:API_KEY="my-secret-key"; then the
     uvicorn command. Without API_KEY the server uses the demo key
     "demo-key-please-change".)
  2. In another terminal, from this same folder:
         API_URL=http://127.0.0.1:8000 API_KEY=my-secret-key python example_client.py
     On Windows PowerShell:
         $env:API_URL="http://127.0.0.1:8000"; $env:API_KEY="my-secret-key"; python example_client.py

After you deploy on Render, point it at your live URL instead:
         API_URL=https://crop-price-api-xxxx.onrender.com API_KEY=<key from Render> python example_client.py
The first call after the free server sleeps takes ~30-60 seconds to wake
up, so the timeout here is 90 seconds — that is normal, not an error.

Prices come back in USD per metric ton. The table also shows PKR per
maund (40 kg): USD/MT x USD_PKR rate / 25. Default rate 276.61 PKR/USD
(the rate used on 2026-11-13); override it with USD_PKR=280.
"""

import os
import sys

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "demo-key-please-change")
USD_PKR = float(os.environ.get("USD_PKR", "276.61"))  # PKR per USD
MAUND_KG = 40  # 1 metric ton = 1000 kg, so 1 maund = 40 kg = 1/25 of a ton


def usd_to_pkr_per_maund(usd_per_ton):
    return usd_per_ton * USD_PKR * MAUND_KG / 1000.0


def get(path, params=None, need_key=True):
    headers = {"X-API-Key": API_KEY} if need_key else {}
    try:
        r = requests.get(f"{API_URL}{path}", params=params,
                         headers=headers, timeout=90)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: could not reach the API at {API_URL}")
        print("Is the server running? Start it with:")
        print("  API_KEY=my-secret-key uvicorn app:app --port 8000")
        sys.exit(1)
    if r.status_code == 401:
        print("ERROR 401: missing or wrong API key.")
        print("Set API_KEY to the same value the server was started with")
        print("(on Render: the API_KEY in the service's Environment tab).")
        sys.exit(1)
    r.raise_for_status()
    return r.json()


def print_forecast(title, data):
    print(f"\n{title}")
    print(f"Model: {data['model']} | Last price: ${data['last_price']} "
          f"({data['last_date']}) | Backtest MAPE: "
          f"{data['backtest']['mape_pct']}%")
    print(f"{'Date':<12} {'USD / ton':>11} {'PKR / maund':>13}")
    for row in data["forecast"]:
        print(f"{row['date']:<12} {row['price']:>11.2f} "
              f"{usd_to_pkr_per_maund(row['price']):>13,.0f}")


def main():
    print(f"API: {API_URL}")
    health = get("/health", need_key=False)
    print("Health:", health)

    daily = get("/predict/daily",
                params={"crop": "wheat", "days": 7, "model": "auto"})
    print_forecast("Wheat — next 7 trading days", daily)

    monthly = get("/predict/monthly",
                  params={"crop": "wheat", "months": 6, "model": "auto"})
    print_forecast("Wheat — next 6 months", monthly)

    print("\nDone. Change crop= to corn (daily) or maize/rice (monthly),")
    print("days= up to 30, months= up to 12, model= to naive /")
    print("linear_regression / xgboost to compare models.")


if __name__ == "__main__":
    main()
