# explore.py
# Step 1: Load and explore the Superstore dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load data ────────────────────────────────────────────
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# ── 2. Basic info ───────────────────────────────────────────
print("=== SHAPE ===")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

print("\n=== COLUMNS & TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== SALES STATISTICS ===")
print(df[["Sales", "Profit", "Quantity", "Discount"]].describe().round(2))

print("\n=== UNIQUE CATEGORIES ===")
for col in ["Category", "Region", "Segment"]:
    print(f"  {col}: {df[col].unique().tolist()}")

# ── 3. Parse dates ──────────────────────────────────────────
df["Order Date"] = pd.to_datetime(df["Order Date"])
print(f"\n=== DATE RANGE ===")
print(f"  From : {df['Order Date'].min().date()}")
print(f"  To   : {df['Order Date'].max().date()}")
print(f"  Months: {df['Order Date'].dt.to_period('M').nunique()}")

# ── 4. Monthly sales ────────────────────────────────────────
monthly = (df.groupby(df["Order Date"].dt.to_period("M"))
             .agg(Sales=("Sales", "sum"),
                  Orders=("Order ID", "nunique"),
                  Profit=("Profit", "sum"))
             .reset_index())
monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()

print("\n=== MONTHLY SALES (first 6 rows) ===")
print(monthly.head(6).to_string(index=False))

# ── 5. Plot monthly trend ────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.bar(monthly["Order Date"], monthly["Sales"],
        color="#378ADD", alpha=0.6, width=20, label="Monthly Sales")
plt.plot(monthly["Order Date"],
         monthly["Sales"].rolling(3).mean(),
         color="#D85A30", linewidth=2, label="3-month rolling avg")
plt.title("Superstore — Monthly Sales Trend (2014–2017)", fontsize=13, fontweight="bold")
plt.ylabel("Sales (USD)")
plt.xlabel("Month")
plt.legend()
plt.tight_layout()
plt.savefig("01_monthly_trend.png", dpi=150)
plt.show()
print("\nChart saved as 01_monthly_trend.png")