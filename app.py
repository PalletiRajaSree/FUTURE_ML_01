# app.py
# Step 5: Interactive Streamlit Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecast | Superstore",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f8f7f4; }
    .metric-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #666; }
    .metric-value { font-size: 26px; font-weight: 600; color: #185FA5; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"])
    df["Sales"]      = df["Sales"].clip(lower=0)
    df["Margin"]     = df["Profit"] / (df["Sales"] + 1e-9)
    return df


def build_monthly(df, category="All", region="All", segment="All"):
    tmp = df.copy()
    if category != "All": tmp = tmp[tmp["Category"] == category]
    if region   != "All": tmp = tmp[tmp["Region"]   == region]
    if segment  != "All": tmp = tmp[tmp["Segment"]  == segment]

    monthly = (tmp.groupby(tmp["Order Date"].dt.to_period("M"))
                  .agg(Sales    = ("Sales",    "sum"),
                       Profit   = ("Profit",   "sum"),
                       Orders   = ("Order ID", "nunique"),
                       Quantity = ("Quantity", "sum"))
                  .reset_index())
    monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()
    monthly = monthly.rename(columns={"Order Date": "date"})
    monthly = monthly.sort_values("date").reset_index(drop=True)

    # Features
    monthly["time_index"] = np.arange(1, len(monthly) + 1)
    monthly["month"]      = monthly["date"].dt.month
    monthly["month_sin"]  = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"]  = np.cos(2 * np.pi * monthly["month"] / 12)
    monthly["quarter"]    = monthly["date"].dt.quarter
    for q in [1, 2, 3, 4]:
        monthly[f"q{q}"] = (monthly["quarter"] == q).astype(int)

    return monthly


def run_linear_regression(monthly, train_pct, horizon):
    series = monthly["Sales"].values
    split  = int(len(series) * train_pct / 100)

    feature_cols = ["time_index", "month_sin", "month_cos",
                    "q1", "q2", "q3", "q4"]
    X = monthly[feature_cols].values
    y = series

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    fitted    = model.predict(X_train_sc)
    test_pred = model.predict(X_test_sc) if len(X_test) > 0 else np.array([])

    # 6-month future forecast
    last_idx   = monthly["time_index"].max()
    last_month = monthly["month"].iloc[-1]
    future_rows = []
    for i in range(1, horizon + 1):
        m = ((last_month - 1 + i) % 12) + 1
        t = last_idx + i
        q = (m - 1) // 3 + 1
        future_rows.append({
            "time_index": t,
            "month_sin": np.sin(2 * np.pi * m / 12),
            "month_cos": np.cos(2 * np.pi * m / 12),
            "q1": int(q==1), "q2": int(q==2),
            "q3": int(q==3), "q4": int(q==4),
        })
    X_future  = scaler.transform(pd.DataFrame(future_rows)[feature_cols].values)
    forecast  = model.predict(X_future)

    residuals = y_train - fitted
    ci        = 1.96 * np.std(residuals)

    metrics = {}
    if len(y_test) > 0:
        metrics["MAE"]  = mean_absolute_error(y_test, test_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, test_pred))
        metrics["MAPE"] = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        metrics["R2"]   = r2_score(y_test, test_pred)

    return {
        "series": series, "split": split,
        "fitted": fitted, "test_pred": test_pred,
        "forecast": forecast, "ci": ci,
        "residuals": residuals, "metrics": metrics,
        "y_train": y_train, "y_test": y_test,
    }


def moving_average_forecast(train, n_steps, window=3):
    history = list(train)
    slope   = np.polyfit(np.arange(len(train)), train, 1)[0]
    preds   = []
    for _ in range(n_steps):
        pred = np.mean(history[-window:]) + slope
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


def exp_smoothing_forecast(train, n_steps, alpha=0.3):
    smoothed = [train[0]]
    for v in train[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    slope = np.polyfit(np.arange(len(train)), train, 1)[0]
    last  = smoothed[-1]
    return np.array([last + slope * (i + 1) for i in range(n_steps)])


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
df = load_data()

st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=55)
st.sidebar.title("Sales Forecast App")
st.sidebar.caption("Superstore · 2014–2017 · ML Task 1")
st.sidebar.divider()

st.sidebar.subheader("⚙️ Filters")
category  = st.sidebar.selectbox("Category",  ["All"] + list(df["Category"].unique()))
region    = st.sidebar.selectbox("Region",    ["All"] + list(df["Region"].unique()))
segment   = st.sidebar.selectbox("Segment",   ["All"] + list(df["Segment"].unique()))

st.sidebar.divider()
st.sidebar.subheader("⚙️ Model Settings")
train_pct  = st.sidebar.slider("Training split (%)", 60, 90, 75, step=5)
horizon    = st.sidebar.slider("Forecast horizon (months)", 3, 12, 6, step=3)

st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📈 Forecast",
    "📊 Model Evaluation",
    "💡 Insights"
])


# ════════════════════════════════════════════════════════════
# BUILD DATA
# ════════════════════════════════════════════════════════════
monthly = build_monthly(df, category, region, segment)
results = run_linear_regression(monthly, train_pct, horizon)

series     = results["series"]
split      = results["split"]
forecast   = results["forecast"]
ci         = results["ci"]
metrics    = results["metrics"]

future_dates = pd.date_range(
    monthly["date"].iloc[-1] + pd.DateOffset(months=1),
    periods=horizon, freq="MS"
)


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📈 Sales & Demand Forecasting")
    st.caption("Superstore dataset · Future Interns ML Task 1 · Sujan Kumar")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue",  f"${df['Sales'].sum()/1e6:.2f}M")
    c2.metric("Total Profit",   f"${df['Profit'].sum()/1e6:.2f}M")
    c3.metric("Total Orders",   f"{df['Order ID'].nunique():,}")
    c4.metric("Profit Margin",  f"{df['Profit'].sum()/df['Sales'].sum()*100:.1f}%")

    st.divider()
    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly["date"], monthly["Sales"],
           color="#378ADD", alpha=0.55, width=20, label="Monthly Sales")
    ax.plot(monthly["date"],
            monthly["Sales"].rolling(3).mean(),
            color="#D85A30", linewidth=2, label="3-mo Rolling Avg")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
    ax.set_xlabel("Month"); ax.set_ylabel("Sales (USD)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Revenue by Category")
        cat = df.groupby("Category")["Sales"].sum().sort_values()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        colors = ["#534AB7", "#1D9E75", "#378ADD"]
        bars = ax2.barh(cat.index, cat.values, color=colors, alpha=0.85)
        for bar, val in zip(bars, cat.values):
            ax2.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                     f"${val/1e6:.2f}M", va="center", fontsize=9)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1e6:.1f}M"))
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout(); st.pyplot(fig2)

    with c2:
        st.subheader("Revenue by Region")
        reg = df.groupby("Region")["Sales"].sum().sort_values()
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        colors2 = ["#D85A30", "#534AB7", "#1D9E75", "#378ADD"]
        bars2 = ax3.barh(reg.index, reg.values, color=colors2, alpha=0.85)
        for bar, val in zip(bars2, reg.values):
            ax3.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                     f"${val/1e6:.2f}M", va="center", fontsize=9)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1e6:.1f}M"))
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        fig3.tight_layout(); st.pyplot(fig3)


# ════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ════════════════════════════════════════════════════════════
elif page == "📈 Forecast":
    st.title("📈 Sales Forecast")
    st.caption(f"Linear Regression · {train_pct}% train · {horizon}-month horizon")
    st.divider()

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",  f"${metrics['MAE']:,.0f}")
        c2.metric("RMSE", f"${metrics['RMSE']:,.0f}")
        c3.metric("MAPE", f"{metrics['MAPE']:.1f}%")
        c4.metric("R²",   f"{metrics['R2']:.4f}")
        st.divider()

    # Forecast chart
    st.subheader("Historical + Forecast")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["date"], series,
            color="#378ADD", linewidth=2,
            marker="o", markersize=3, label="Actual Sales")

    # Fitted line on training
    train_dates = monthly["date"].values[:split]
    ax.plot(train_dates, results["fitted"],
            color="#1D9E75", linewidth=1.5,
            linestyle="--", alpha=0.8, label="Fitted (train)")

    # Forecast
    ax.plot(future_dates, forecast,
            color="#D85A30", linewidth=2.5,
            linestyle="--", marker="^", markersize=7, label="Forecast")
    ax.fill_between(future_dates,
                    forecast - ci, forecast + ci,
                    color="#BA7517", alpha=0.15, label="95% CI")
    ax.axvline(x=monthly["date"].iloc[-1],
               color="gray", linewidth=1, linestyle=":")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
    ax.set_xlabel("Month"); ax.set_ylabel("Sales (USD)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); st.pyplot(fig)

    # Forecast table
    st.subheader("Forecast Table")
    fc_df = pd.DataFrame({
        "Month":         [d.strftime("%b %Y") for d in future_dates],
        "Forecast ($)":  [f"${v:,.0f}" for v in forecast],
        "Low CI ($)":    [f"${v:,.0f}" for v in forecast - ci],
        "High CI ($)":   [f"${v:,.0f}" for v in forecast + ci],
        "MoM Change":    ["—" if i == 0
                          else f"{(forecast[i]-forecast[i-1])/forecast[i-1]*100:+.1f}%"
                          for i in range(horizon)],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # Residuals
    st.subheader("Residual Errors (Training)")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    res = results["residuals"]
    res_colors = ["#1D9E75" if r >= 0 else "#D85A30" for r in res]
    axes[0].bar(train_dates, res, color=res_colors, width=20, alpha=0.85)
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].set_title("Residuals by Month", fontweight="bold")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].hist(res, bins=10, color="#534AB7", alpha=0.75, edgecolor="white")
    axes[1].axvline(np.mean(res), color="#D85A30", linewidth=1.5,
                    linestyle="--", label=f"Mean: ${np.mean(res):,.0f}")
    axes[1].set_title("Error Distribution", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig2.tight_layout(); st.pyplot(fig2)


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation")
    st.caption("Comparing all three models on the same test set")
    st.divider()

    train = series[:split]
    test  = series[split:]

    # Run all 3 models on test set
    lr_pred = results["test_pred"]
    ma_pred = moving_average_forecast(train, len(test), window=3)
    es_pred = exp_smoothing_forecast(train, len(test), alpha=0.3)

    def get_metrics(actual, pred):
        if len(pred) == 0: return {}
        return {
            "MAE":  mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "MAPE": np.mean(np.abs((actual - pred) / actual)) * 100,
            "R2":   r2_score(actual, pred),
        }

    all_metrics = {
        "Linear Regression": get_metrics(test, lr_pred),
        "Moving Average":    get_metrics(test, ma_pred),
        "Exp. Smoothing":    get_metrics(test, es_pred),
    }

    # Table
    rows = []
    for name, m in all_metrics.items():
        if m:
            rows.append({
                "Model":    name,
                "MAE ($)":  f"${m['MAE']:,.0f}",
                "RMSE ($)": f"${m['RMSE']:,.0f}",
                "MAPE (%)": f"{m['MAPE']:.1f}%",
                "R²":       f"{m['R2']:.4f}",
            })

    best_name = min(all_metrics, key=lambda k: all_metrics[k].get("MAE", 9e9))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.success(f"✅ Best model: **{best_name}** — lowest MAE on the test set")
    st.divider()

    # Comparison charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    test_dates = monthly["date"].values[split:]

    # Actual vs predicted
    axes[0].plot(test_dates, test,    color="#378ADD", linewidth=2,   label="Actual", marker="o", markersize=5)
    if len(lr_pred) > 0:
        axes[0].plot(test_dates, lr_pred, color="#1D9E75", linewidth=1.8, label="Lin. Reg.",  linestyle="--")
    axes[0].plot(test_dates, ma_pred, color="#534AB7", linewidth=1.8, label="Mov. Avg.",  linestyle="-.")
    axes[0].plot(test_dates, es_pred, color="#D85A30", linewidth=1.8, label="Exp. Smooth",linestyle=":")
    axes[0].set_title("Actual vs All Models (test set)", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(True, alpha=0.3)

    # MAE bar
    names = list(all_metrics.keys())
    maes  = [all_metrics[n]["MAE"] for n in names]
    colors = ["#378ADD", "#534AB7", "#D85A30"]
    bars = axes[1].bar(names, maes, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() * 1.02,
                     f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("MAE Comparison (lower = better)", fontweight="bold")
    axes[1].set_ylabel("MAE ($)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout(); st.pyplot(fig)

    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | Meaning |
|---|---|
| **MAE** | Average dollar error per month — most intuitive |
| **RMSE** | Penalises large errors more than MAE |
| **MAPE** | % error — under 10% is considered good for business forecasting |
| **R²** | 1.0 = perfect fit · above 0.85 = strong model |
        """)


# ════════════════════════════════════════════════════════════
# PAGE 4 — INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.title("💡 Business Insights")
    st.caption("Actionable findings from the Superstore forecast model")
    st.divider()

    growth = (series[-1] - series[0]) / series[0] * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("4-year growth",        f"{growth:.1f}%")
    c2.metric("6-mo forecast total",  f"${forecast.sum()/1e6:.2f}M")
    c3.metric("Peak month ever",      f"${series.max()/1000:.0f}K")

    st.divider()

    insights = [
        ("📈 Strong upward trend",
         f"Revenue grew **{growth:.0f}%** over 4 years — roughly $1,500 more per month on average. "
         "The linear model captures this well with R² above 0.85."),
        ("🎄 Q4 is your peak season",
         "November and December spike above 1.4× the annual average every year. "
         "Plan inventory and marketing **6–8 weeks in advance** to capture peak demand."),
        ("❌ Tables & Bookcases lose money",
         "Despite strong sales volume, Tables lose **$17,725** and Bookcases **$3,473** in total profit. "
         "Excessive discounting is the cause — review discount policy immediately."),
        ("⚠️ Discounts above 20% kill profit",
         "Average profit at 0% discount = **$66.90**. At 21–40% discount = **-$77.86**. "
         "Discounting beyond 20% directly destroys margin."),
        ("🌍 West region leads",
         "The West generates the highest regional revenue. Central lags behind — "
         "a targeted regional campaign could unlock significant untapped demand."),
        ("🎯 Use upper CI for inventory",
         "Always procure stock based on the **upper confidence bound** to avoid stockouts. "
         "Use the lower bound for conservative cash flow planning."),
    ]

    for i in range(0, len(insights), 2):
        c1, c2 = st.columns(2)
        for col, (title, body) in zip([c1, c2], insights[i:i+2]):
            with col:
                st.markdown(f"""
<div style="background:#f0f9f4;border-left:4px solid #1D9E75;
            padding:14px 16px;border-radius:6px;margin-bottom:12px">
  <strong>{title}</strong><br><br>
  <span style="font-size:14px;line-height:1.6">{body}</span>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📅 6-month forecast table")
    fc_df = pd.DataFrame({
        "Month":        [d.strftime("%B %Y") for d in future_dates],
        "Forecast ($)": [f"${v:,.0f}" for v in forecast],
        "Low CI ($)":   [f"${v:,.0f}" for v in forecast - ci],
        "High CI ($)":  [f"${v:,.0f}" for v in forecast + ci],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)