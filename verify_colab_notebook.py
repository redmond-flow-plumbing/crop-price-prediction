"""Verify crop_price_prediction_colab.ipynb by executing its code cells
locally (skipping the !pip install cell) and checking the scoreboard
against the verified results from crop_price_prediction.py."""
import json
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

nb = json.load(open("/home/hatch/workspace/your_files/crop-price-prediction/crop_price_prediction_colab.ipynb"))
ns = {"__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if any(line.strip().startswith("!") for line in src.splitlines()):
        print(f"[cell {i}] skipped shell/pip cell")
        continue
    print(f"[cell {i}] executing ({len(src.splitlines())} lines)...")
    exec(compile(src, f"<cell {i}>", "exec"), ns)

# Check scoreboards captured in all_results
expected = {
    "wheat": ("XGBoost", 14.81, 5.68),
    "maize": ("Linear Regression", 10.49, 3.48),
    "rice": ("Linear Regression", 22.36, 4.17),
}
ok = True
for crop, (exp_model, exp_rmse, exp_mape) in expected.items():
    name, score, best, *_ = ns["all_results"][crop]
    row = score[score["model"] == exp_model].iloc[0]
    rmse_ok = abs(row["RMSE"] - exp_rmse) < 0.05
    mape_ok = abs(row["MAPE_%"] - exp_mape) < 0.05
    best_ok = best == exp_model
    status = "OK" if (rmse_ok and mape_ok and best_ok) else "MISMATCH"
    if status != "OK":
        ok = False
    print(f"{name}: best={best} (expect {exp_model}) | "
          f"{exp_model} RMSE={row['RMSE']} (expect {exp_rmse}) "
          f"MAPE={row['MAPE_%']}% (expect {exp_mape}) -> {status}")

print("NOTEBOOK VERIFICATION:", "PASS — all 3 crops match verified results" if ok else "FAIL")
sys.exit(0 if ok else 1)
