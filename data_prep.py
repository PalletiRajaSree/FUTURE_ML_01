# data_prep.py
# Step 3: Data Cleaning & Feature Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load & Clean ─────────────────────────────────────────
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Parse dates
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"]  = pd.to_datetime(df["Ship Date"])

# Remove duplicates (if any)
before = len(df)
df = df.drop_duplicates()
print(f"Duplicates removed: {before - len(df)}")

# Clip negative sales (shouldn't exist)
df["Sales"]    = df["Sales"].clip(lower=0)
df["Quantity"] = df["Quantity"].clip(lower=0)

# New derived column — profit margin per order
df["Margin"] = df["Profit"] / (df["Sales"] + 1e-9)

print(f"Clean dataset: {len(df)} rows")
print(df[["Order Date", "Sales", "Profit", "Margin"]].head(5).to_string(index=False))


# ── 2. Aggregate to Monthly ──────────────────────────────────
monthly = (df.groupby(df["Order Date"].dt.to_period("M"))
             .agg(Sales    = ("Sales",    "sum"),
                  Profit   = ("Profit",   "sum"),
                  Orders   = ("Order ID", "nunique"),
                  Quantity = ("Quantity", "sum"))
             .reset_index())

monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()
monthly = monthly.rename(columns={"Order Date": "date"})
monthly = monthly.sort_values("date").reset_index(drop=True)

print(f"\nMonthly data: {len(monthly)} rows")
print(monthly.head(6).to_string(index=False))


# ── 3. Feature Engineering ───────────────────────────────────

# A) Ordinal time index (1, 2, 3 ... 48)
monthly["time_index"] = np.arange(1, len(monthly) + 1)

# B) Month number
monthly["month"] = monthly["date"].dt.month

# C) Cyclical encoding — captures Jan is close to Dec
monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)

# D) Quarter flags
monthly["quarter"] = monthly["date"].dt.quarter
for q in [1, 2, 3, 4]:
    monthly[f"q{q}"] = (monthly["quarter"] == q).astype(int)

# E) Lag features (past values as predictors)
monthly["lag_1"]  = monthly["Sales"].shift(1)   # last month
monthly["lag_3"]  = monthly["Sales"].shift(3)   # 3 months ago
monthly["lag_12"] = monthly["Sales"].shift(12)  # same month last year

# F) Rolling averages (trend smoothing)
monthly["roll_3"] = monthly["Sales"].rolling(3).mean().shift(1)
monthly["roll_6"] = monthly["Sales"].rolling(6).mean().shift(1)

print("\n=== ENGINEERED FEATURES (last 6 rows) ===")
print(monthly[["date","Sales","time_index","month_sin","month_cos",
               "lag_1","lag_12","roll_3"]].tail(6).to_string(index=False))


# ── 4. Save cleaned data ─────────────────────────────────────
monthly.to_csv("monthly_sales.csv", index=False)
print("\nSaved → monthly_sales.csv")


# ── 5. Visualize features ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Sales + rolling avg
axes[0,0].plot(monthly["date"], monthly["Sales"],
               color="#378ADD", linewidth=1.8, label="Monthly Sales")
axes[0,0].plot(monthly["date"], monthly["roll_3"],
               color="#D85A30", linewidth=1.5, linestyle="--", label="3-mo Rolling")
axes[0,0].set_title("Sales + Rolling Average", fontweight="bold")
axes[0,0].legend(fontsize=9)
axes[0,0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))

# Lag 1 vs Sales scatter
axes[0,1].scatter(monthly["lag_1"], monthly["Sales"],
                  color="#534AB7", alpha=0.6, s=40)
axes[0,1].set_title("Lag-1 vs Current Sales", fontweight="bold")
axes[0,1].set_xlabel("Sales last month")
axes[0,1].set_ylabel("Sales this month")

# Seasonal index
avg_by_month = monthly.groupby("month")["Sales"].mean()
overall_avg  = avg_by_month.mean()
seasonal_idx = avg_by_month / overall_avg
month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
colors = ["#534AB7" if v >= 1 else "#D0CCF0" for v in seasonal_idx]
axes[1,0].bar(month_names, seasonal_idx.values, color=colors, alpha=0.9)
axes[1,0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
axes[1,0].set_title("Seasonal Index by Month", fontweight="bold")
axes[1,0].set_ylabel("Index (1.0 = average)")

# Time index vs Sales (linear trend check)
axes[1,1].scatter(monthly["time_index"], monthly["Sales"],
                  color="#1D9E75", alpha=0.6, s=40)
m, b = np.polyfit(monthly["time_index"], monthly["Sales"], 1)
x = monthly["time_index"]
axes[1,1].plot(x, m * x + b, color="#D85A30", linewidth=2, label=f"Trend: +${m:.0f}/mo")
axes[1,1].set_title("Linear Trend over Time", fontweight="bold")
axes[1,1].set_xlabel("Month number (1–48)")
axes[1,1].legend(fontsize=9)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

fig.suptitle("Feature Engineering Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("02_features.png", dpi=150)
plt.show()
print("Chart saved as 02_features.png")