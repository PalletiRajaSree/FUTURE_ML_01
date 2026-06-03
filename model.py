# model.py
# Step 4: Build, train and forecast with 3 models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load cleaned monthly data ─────────────────────────────
monthly = pd.read_csv("monthly_sales.csv", parse_dates=["date"])
monthly = monthly.dropna()  # drop rows with NaN lags (first 12 rows)

print(f"Data loaded: {len(monthly)} months after dropping NaN lags")

series = monthly["Sales"].values
dates  = monthly["date"].values


# ── 2. Train / Test Split (75% train, 25% test) ───────────────
split      = int(len(series) * 0.75)
train_data = series[:split]
test_data  = series[split:]
train_dates = dates[:split]
test_dates  = dates[split:]

print(f"Train: {split} months | Test: {len(test_data)} months")


# ════════════════════════════════════════════════════════════
# MODEL 1 — Linear Regression
# ════════════════════════════════════════════════════════════
print("\n--- MODEL 1: Linear Regression ---")

# Features for regression
feature_cols = ["time_index", "month_sin", "month_cos", "q1", "q2", "q3", "q4"]
X = monthly[feature_cols].values
y = series

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale features
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train
lr_model = LinearRegression()
lr_model.fit(X_train_sc, y_train)

# Predict on test set
lr_pred = lr_model.predict(X_test_sc)

# Metrics
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mape = np.mean(np.abs((y_test - lr_pred) / y_test)) * 100
lr_r2   = r2_score(y_test, lr_pred)

print(f"  MAE  : ${lr_mae:,.0f}")
print(f"  RMSE : ${lr_rmse:,.0f}")
print(f"  MAPE : {lr_mape:.1f}%")
print(f"  R²   : {lr_r2:.4f}")


# ════════════════════════════════════════════════════════════
# MODEL 2 — Moving Average
# ════════════════════════════════════════════════════════════
print("\n--- MODEL 2: Moving Average (window=3) ---")

WINDOW = 3

def moving_average_forecast(train, n_steps, window=3):
    """Forecast n_steps ahead using rolling mean + linear drift."""
    history = list(train)
    # linear drift from training data
    x = np.arange(len(train))
    slope = np.polyfit(x, train, 1)[0]
    preds = []
    for i in range(n_steps):
        pred = np.mean(history[-window:]) + slope
        preds.append(pred)
        history.append(pred)
    return np.array(preds)

ma_pred = moving_average_forecast(train_data, len(test_data), window=WINDOW)

ma_mae  = mean_absolute_error(y_test, ma_pred)
ma_rmse = np.sqrt(mean_squared_error(y_test, ma_pred))
ma_mape = np.mean(np.abs((y_test - ma_pred) / y_test)) * 100
ma_r2   = r2_score(y_test, ma_pred)

print(f"  MAE  : ${ma_mae:,.0f}")
print(f"  RMSE : ${ma_rmse:,.0f}")
print(f"  MAPE : {ma_mape:.1f}%")
print(f"  R²   : {ma_r2:.4f}")


# ════════════════════════════════════════════════════════════
# MODEL 3 — Exponential Smoothing
# ════════════════════════════════════════════════════════════
print("\n--- MODEL 3: Exponential Smoothing (alpha=0.3) ---")

def exp_smoothing_forecast(train, n_steps, alpha=0.3):
    """Single exponential smoothing with linear trend extrapolation."""
    # Smooth the training series
    smoothed = [train[0]]
    for v in train[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    # Linear trend on training data
    x     = np.arange(len(train))
    slope = np.polyfit(x, train, 1)[0]
    last  = smoothed[-1]
    preds = [last + slope * (i + 1) for i in range(n_steps)]
    return np.array(preds)

es_pred = exp_smoothing_forecast(train_data, len(test_data), alpha=0.3)

es_mae  = mean_absolute_error(y_test, es_pred)
es_rmse = np.sqrt(mean_squared_error(y_test, es_pred))
es_mape = np.mean(np.abs((y_test - es_pred) / y_test)) * 100
es_r2   = r2_score(y_test, es_pred)

print(f"  MAE  : ${es_mae:,.0f}")
print(f"  RMSE : ${es_rmse:,.0f}")
print(f"  MAPE : {es_mape:.1f}%")
print(f"  R²   : {es_r2:.4f}")


# ════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ════════════════════════════════════════════════════════════
print("\n=== MODEL COMPARISON ===")
print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'R²':>8}")
print("-" * 65)
models = [
    ("Linear Regression", lr_mae, lr_rmse, lr_mape, lr_r2),
    ("Moving Average",    ma_mae, ma_rmse, ma_mape, ma_r2),
    ("Exp Smoothing",     es_mae, es_rmse, es_mape, es_r2),
]
for name, mae, rmse, mape, r2 in models:
    print(f"{name:<25} ${mae:>9,.0f} ${rmse:>9,.0f} {mape:>7.1f}% {r2:>8.4f}")

best = min(models, key=lambda x: x[1])
print(f"\n✅ Best model: {best[0]} (lowest MAE)")


# ════════════════════════════════════════════════════════════
# FORECAST NEXT 6 MONTHS
# ════════════════════════════════════════════════════════════
print("\n=== 6-MONTH FORECAST (Linear Regression) ===")

HORIZON = 6
last_idx   = monthly["time_index"].max()
last_month = monthly["month"].iloc[-1]

future_rows = []
for i in range(1, HORIZON + 1):
    m = ((last_month - 1 + i) % 12) + 1
    t = last_idx + i
    q = (m - 1) // 3 + 1
    future_rows.append({
        "time_index": t,
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
        "q1": int(q == 1), "q2": int(q == 2),
        "q3": int(q == 3), "q4": int(q == 4),
    })

future_df = pd.DataFrame(future_rows)
X_future  = scaler.transform(future_df[feature_cols].values)
forecast  = lr_model.predict(X_future)

# Confidence interval (±1.96 * residual std)
residuals = y_train - lr_model.predict(X_train_sc)
ci        = 1.96 * np.std(residuals)

future_dates = pd.date_range(
    monthly["date"].iloc[-1] + pd.DateOffset(months=1),
    periods=HORIZON, freq="MS"
)
print(f"{'Month':<12} {'Forecast':>12} {'Low (95%)':>12} {'High (95%)':>12}")
print("-" * 50)
for d, f in zip(future_dates, forecast):
    print(f"{d.strftime('%b %Y'):<12} ${f:>11,.0f} ${f-ci:>11,.0f} ${f+ci:>11,.0f}")


# ════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# ── Chart 1: Actual vs Predicted (all 3 models on test set) ──
ax = axes[0, 0]
ax.plot(test_dates, y_test,   color="#378ADD", linewidth=2,   label="Actual",     marker="o", markersize=4)
ax.plot(test_dates, lr_pred,  color="#1D9E75", linewidth=1.8, label="Lin. Reg.",  linestyle="--")
ax.plot(test_dates, ma_pred,  color="#534AB7", linewidth=1.8, label="Mov. Avg.",  linestyle="-.")
ax.plot(test_dates, es_pred,  color="#D85A30", linewidth=1.8, label="Exp. Smooth",linestyle=":")
ax.set_title("Test Set: Actual vs All Models", fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── Chart 2: 6-month Forecast ────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(dates, series, color="#378ADD", linewidth=2, label="Historical", marker="o", markersize=3)
ax2.plot(future_dates, forecast, color="#D85A30", linewidth=2.5,
         linestyle="--", marker="^", markersize=7, label="Forecast")
ax2.fill_between(future_dates, forecast - ci, forecast + ci,
                 color="#BA7517", alpha=0.15, label="95% CI")
ax2.axvline(x=dates[-1], color="gray", linewidth=1, linestyle=":")
ax2.set_title("6-Month Sales Forecast", fontweight="bold")
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
ax2.grid(True, alpha=0.3)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# ── Chart 3: Residuals ───────────────────────────────────────
ax3 = axes[1, 0]
residuals_test = y_test - lr_pred
colors = ["#1D9E75" if r >= 0 else "#D85A30" for r in residuals_test]
ax3.bar(test_dates, residuals_test, color=colors, width=20, alpha=0.85)
ax3.axhline(0, color="gray", linewidth=1)
ax3.set_title("Residual Errors (Linear Regression)", fontweight="bold")
ax3.set_ylabel("Actual − Predicted")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
ax3.grid(True, alpha=0.3)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# ── Chart 4: Model Comparison Bar ────────────────────────────
ax4 = axes[1, 1]
model_names = ["Linear Reg.", "Moving Avg.", "Exp. Smooth"]
maes  = [lr_mae, ma_mae, es_mae]
mapes = [lr_mape, ma_mape, es_mape]
x = np.arange(len(model_names))
w = 0.35
b1 = ax4.bar(x - w/2, maes,  width=w, color="#378ADD", alpha=0.85, label="MAE ($)")
b2 = ax4.bar(x + w/2, mapes, width=w, color="#534AB7", alpha=0.85, label="MAPE (%)")
ax4.set_xticks(x); ax4.set_xticklabels(model_names, fontsize=9)
ax4.set_title("Model Comparison", fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

fig.suptitle("Sales Forecast — Model Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("03_model_results.png", dpi=150)
plt.show()
print("\nChart saved as 03_model_results.png")