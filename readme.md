# 📈 Sales & Demand Forecasting — Superstore
**Future Interns · Machine Learning Task 1 (2026)**
**Author: Sujan Kumar**

---

## 🧾 Project Overview

This project builds a complete **Sales & Demand Forecasting system** using the Superstore retail dataset.
It covers the full ML pipeline — from raw data to an interactive business dashboard — satisfying every requirement of the Future Interns ML Task 1.

The goal is not just to train a model, but to deliver **business-ready insights** that a store owner, startup founder, or business manager can act on directly.

---

## 🎯 What the Project Does

- Loads and cleans 4 years of real Superstore transaction data (9,994 orders)
- Aggregates daily orders into a monthly time series
- Engineers 8 time-based features including lag, rolling averages, and cyclical encoding
- Trains and compares 3 forecasting models
- Forecasts the next 6 months with a 95% confidence interval
- Presents everything in a 4-page interactive Streamlit dashboard

---

## 📁 Project Structure

```
sales_forecast/
│
├── app.py                        # Streamlit dashboard (4 pages)
├── explore.py                    # Step 1 — data loading & exploration
├── data_prep.py                  # Step 2 — cleaning & feature engineering
├── model.py                      # Step 3 — model training & evaluation
│
├── Sample - Superstore.csv       # Raw dataset (input)
├── monthly_sales.csv             # Cleaned monthly aggregated data (output)
│
├── 01_monthly_trend.png          # Chart — monthly sales trend
├── 02_features.png               # Chart — engineered features overview
├── 03_model_results.png          # Chart — model comparison & forecast
│
└── requirements.txt              # Python dependencies
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11+ | Core language |
| Pandas | Data loading, cleaning, aggregation |
| NumPy | Numerical operations, feature engineering |
| Scikit-learn | Linear Regression, StandardScaler, metrics |
| Matplotlib | All charts and visualizations |
| Streamlit | Interactive web dashboard |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sales-forecast.git
cd sales-forecast
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the scripts in order (optional)
```bash
python explore.py      # Step 1 — explore the data
python data_prep.py    # Step 2 — clean & engineer features
python model.py        # Step 3 — train models & see results
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Name | Sample - Superstore |
| Source | [Kaggle — Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) |
| Rows | 9,994 order line items |
| Period | January 2014 – December 2017 (48 months) |
| Key columns | Order Date, Sales, Profit, Category, Region, Segment |

---

## ⚙️ Feature Engineering

Eight time-based features are engineered from the raw dates:

| Feature | Type | Description |
|---|---|---|
| `time_index` | Ordinal | Sequential month number 1 → 48 (captures linear trend) |
| `month_sin` | Cyclical | Sine encoding of month (Jan close to Dec) |
| `month_cos` | Cyclical | Cosine encoding of month |
| `q1 – q4` | Dummy | Quarter one-hot flags for seasonal patterns |
| `lag_1` | Lag | Sales from 1 month prior |
| `lag_3` | Lag | Sales from 3 months prior |
| `lag_12` | Lag | Sales from same month last year (YoY) |
| `roll_3` | Rolling | 3-month rolling average (shifted by 1) |
| `roll_6` | Rolling | 6-month rolling average (shifted by 1) |

---

## 🤖 Models Implemented

### 1. Linear Regression
Uses `time_index`, `month_sin`, `month_cos`, and quarter flags as features.
Features are scaled with `StandardScaler` before fitting.
Best overall performance with R² above 0.85.

### 2. Moving Average
3-month rolling window with a linear drift component.
Simple and interpretable — good baseline model.

### 3. Exponential Smoothing
Weighted recent observations (α = 0.3) with linear trend extrapolation.
Gives more weight to recent months than older data.

---

## 📐 Model Evaluation

Each model is evaluated on a **25% held-out test set** using four metrics:

| Metric | Description | Target |
|---|---|---|
| MAE | Mean Absolute Error — average dollar error per month | Lower is better |
| RMSE | Root Mean Squared Error — penalises large errors more | Lower is better |
| MAPE | Mean Absolute Percentage Error | Under 10% = good |
| R² | Coefficient of Determination | Above 0.85 = strong |

---

## 📈 App Pages

The Streamlit dashboard has 4 pages accessible from the sidebar:

### 🏠 Overview
- Total revenue, profit, orders, and profit margin KPIs
- Monthly sales trend with 3-month rolling average
- Revenue breakdown by Category and Region

### 📈 Forecast
- Interactive forecast chart with historical data and 6-month projection
- 95% confidence interval band
- Forecast table with month-by-month values and MoM % change
- Residual error bar chart and error distribution histogram

### 📊 Model Evaluation
- Side-by-side metrics table for all 3 models
- Actual vs predicted chart on the test set
- MAE comparison bar chart
- Metric definitions explained in plain English

### 💡 Business Insights
- 6 actionable business findings derived from the data
- Recommendations on inventory, discounts, regions, and seasonality
- 6-month forward forecast table with confidence bounds

**Sidebar controls** let you filter by Category, Region, and Segment, adjust the training split (60–90%), and change the forecast horizon (3–12 months). All charts update live.

---

## 🔍 Key Findings

1. **Revenue grew 86.6% over 4 years** — consistent upward trend of ~$1,500 per month
2. **Q4 peaks at 1.4× the annual average** — November is the strongest month every year
3. **Tables and Bookcases lose money** — $17,725 and $3,473 in negative profit despite high sales volume
4. **Discounts above 20% flip profit negative** — avg profit at 0% discount = $66.90, at 21–40% = -$77.86
5. **West region leads revenue** — Central lags and is an untapped growth opportunity
6. **Linear Regression is the best model** — achieves MAPE under 10% and R² above 0.85

---

## 📤 Deliverables

- ✅ Sales forecasting model (Linear Regression — best performer)
- ✅ Clear visualizations (3 saved charts + interactive Streamlit app)
- ✅ 6-month forward forecast with 95% confidence interval
- ✅ Business-ready insights presented in plain English
- ✅ Presentable to a store owner, startup founder, or business manager

---

## 🚀 Future Improvements

- Add Facebook Prophet model for automatic seasonality handling
- Build separate models per product category
- Add CSV export button for the forecast table
- Deploy the Streamlit app to Streamlit Cloud (free hosting)
- Add email alert when forecast exceeds a threshold

---

## 📜 License

This project was built as part of the **Future Interns ML Internship Program (2026)**.
Free to use for learning and portfolio purposes.

---

## 🙏 Acknowledgements

- Dataset: [Superstore Sales Dataset — Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- Program: [Future Interns](https://futureinterns.com)
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit