"""
app.py  –  Streamlit Frontend
------------------------------
A fully interactive Python dashboard for bike sales prediction.
Run with:  streamlit run app.py
Communicates with the FastAPI backend at http://localhost:8000
"""
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🏍️ Bike Sales India – 2027 Forecast",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #0f3460;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 4px solid #e94560;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    [data-testid="stSidebar"] { background: #0d0d1a; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/motorcycle.png", width=80)
    st.markdown("## 🏍️ Bike Sales India")
    st.markdown("**2027 Sales Forecasting**")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload bike_sales_india.csv",
        type=["csv"],
        help="Upload the Kaggle India bike sales CSV to train the models.",
    )

    if uploaded_file:
        if st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Uploading & training models … (this may take 1–2 minutes)"):
                resp = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                )
            if resp.status_code == 200:
                result = resp.json()
                st.session_state["trained"] = True
                st.session_state["train_result"] = result
                st.success(f"✅ Trained on {result['rows']:,} rows!")
                st.json(result["metrics"])
            else:
                st.error(f"❌ {resp.json().get('detail', 'Unknown error')}")

    st.markdown("---")
    horizon = st.slider(
        "🔮 Forecast horizon (months)",
        min_value=12, max_value=48, value=36, step=3,
        help="Number of months to forecast into the future (36 = ~2027).",
    )

    st.markdown("---")
    st.markdown("### 🔗 API")
    st.markdown(f"[Swagger UI]({API_BASE}/docs)", unsafe_allow_html=True)
    st.markdown(f"[Health check]({API_BASE}/health)", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: check backend health
# ---------------------------------------------------------------------------
@st.cache_data(ttl=5)
def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable", "models_trained": False}


@st.cache_data(ttl=30)
def get_summary():
    r = requests.get(f"{API_BASE}/summary")
    return r.json() if r.status_code == 200 else {}


@st.cache_data(ttl=30)
def get_forecast(horizon: int):
    r = requests.get(f"{API_BASE}/forecast", params={"horizon": horizon})
    return r.json() if r.status_code == 200 else {}


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🏍️ Bike Sales India — 2027 Prediction Dashboard")
st.caption("Machine Learning powered forecasting using Prophet & XGBoost")

# Health badge
health = check_health()
if health["status"] == "unreachable":
    st.error("⚠️  Backend is not running. Start it with: `uvicorn backend.main:app --reload`")
    st.stop()

if not health.get("models_trained") and not st.session_state.get("trained"):
    st.info("👈  Upload `bike_sales_india.csv` in the sidebar and click **Train Models** to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
summary = get_summary()
forecast_data = get_forecast(horizon)

# Guard
if not summary or not forecast_data:
    st.warning("Could not fetch data from the backend.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)

total_units = summary.get("total_units_sold", 0)
year_range = summary.get("year_range", ["-", "-"])
n_records = summary.get("total_records", 0)

# Compute 2027 projection from prophet forecast
prophet_fc = pd.DataFrame(forecast_data.get("prophet_forecast", []))
projected_2027 = 0
if not prophet_fc.empty:
    fc_2027 = prophet_fc[prophet_fc["ds"].str.startswith("2027")]
    projected_2027 = int(fc_2027["yhat"].clip(lower=0).sum()) if not fc_2027.empty else 0

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_units:,.0f}</div>
        <div class="metric-label">Total Units Sold (Historical)</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{year_range[0]}–{year_range[1]}</div>
        <div class="metric-label">Data Coverage</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_records:,}</div>
        <div class="metric-label">Total Records</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{projected_2027:,.0f}</div>
        <div class="metric-label">Projected Units — 2027</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Analysis", "🏆 Brand Insights", "🤖 Model Metrics"])

# ============================================================
# TAB 1 – FORECAST
# ============================================================
with tab1:
    st.markdown('<div class="section-title">🔮 Sales Forecast to 2027</div>', unsafe_allow_html=True)

    monthly_hist = pd.DataFrame(summary.get("monthly_history", []))
    prophet_full = pd.DataFrame(forecast_data.get("prophet_full", []))
    xgb_fc = pd.DataFrame(forecast_data.get("xgboost_forecast", []))

    if not monthly_hist.empty and not prophet_full.empty:
        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=monthly_hist["date"], y=monthly_hist["units_sold"],
            name="Historical Sales",
            mode="lines",
            line=dict(color="#4fc3f7", width=2),
        ))

        # Prophet forecast + confidence band
        hist_end = monthly_hist["date"].max()
        p_future = prophet_full[prophet_full["ds"] > hist_end]

        if not p_future.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([p_future["ds"], p_future["ds"].iloc[::-1]]),
                y=pd.concat([p_future["yhat_upper"], p_future["yhat_lower"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(233, 69, 96, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Prophet 95% CI",
            ))
            fig.add_trace(go.Scatter(
                x=p_future["ds"], y=p_future["yhat"].clip(lower=0),
                name="Prophet Forecast",
                mode="lines",
                line=dict(color="#e94560", width=2.5, dash="dash"),
            ))

        # XGBoost forecast
        if not xgb_fc.empty:
            fig.add_trace(go.Scatter(
                x=xgb_fc["ds"], y=xgb_fc["yhat"].clip(lower=0),
                name="XGBoost Forecast",
                mode="lines",
                line=dict(color="#ffd700", width=2.5, dash="dot"),
            ))

        # 2027 highlight band
        fig.add_vrect(x0="2027-01", x1="2027-12",
                      fillcolor="rgba(255,215,0,0.05)",
                      layer="below", line_width=0,
                      annotation_text="2027", annotation_position="top left")

        fig.update_layout(
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Month",
            yaxis_title="Units Sold",
            title="Monthly Bike Sales — Historical + Forecast",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Side-by-side 2027 tables
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Prophet 2027 Monthly Forecast**")
        if not prophet_fc.empty:
            fc_display = prophet_fc[prophet_fc["ds"].str.startswith("2027")].copy()
            if not fc_display.empty:
                fc_display["yhat"] = fc_display["yhat"].clip(lower=0).round(0).astype(int)
                if "yhat_lower" in fc_display.columns:
                    fc_display["yhat_lower"] = fc_display["yhat_lower"].clip(lower=0).round(0).astype(int)
                    fc_display["yhat_upper"] = fc_display["yhat_upper"].clip(lower=0).round(0).astype(int)
                st.dataframe(fc_display.rename(columns={
                    "ds": "Month", "yhat": "Forecast", "yhat_lower": "Lower CI", "yhat_upper": "Upper CI"
                }), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("**XGBoost 2027 Monthly Forecast**")
        if not xgb_fc.empty:
            x_display = xgb_fc[xgb_fc["ds"].str.startswith("2027")].copy()
            if not x_display.empty:
                x_display["yhat"] = x_display["yhat"].round(0).astype(int)
                st.dataframe(x_display.rename(columns={"ds": "Month", "yhat": "Forecast"}),
                             use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 – ANALYSIS
# ============================================================
with tab2:
    st.markdown('<div class="section-title">📊 Historical Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Yearly trend
    yearly = pd.DataFrame(summary.get("yearly_trend", []))
    if not yearly.empty:
        with col1:
            fig_yr = px.bar(
                yearly, x="year", y="units_sold",
                title="Yearly Sales Trend",
                color="units_sold",
                color_continuous_scale="Reds",
                template="plotly_dark",
            )
            fig_yr.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_yr, use_container_width=True)

    # Category breakdown
    cat_df = pd.DataFrame(summary.get("categories", []))
    if not cat_df.empty:
        with col2:
            fig_cat = px.pie(
                cat_df, names="category", values="units_sold",
                title="Sales by Category",
                color_discrete_sequence=px.colors.sequential.RdBu,
                template="plotly_dark",
                hole=0.4,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    # State heatmap
    state_df = pd.DataFrame(summary.get("top_states", []))
    if not state_df.empty:
        st.markdown('<div class="section-title">🗺️ Sales by State (Top 10)</div>', unsafe_allow_html=True)
        fig_state = px.bar(
            state_df.head(10), x="units_sold", y="state",
            orientation="h",
            title="Top 10 States by Units Sold",
            color="units_sold",
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig_state.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_state, use_container_width=True)

    # Monthly seasonality
    monthly_hist2 = pd.DataFrame(summary.get("monthly_history", []))
    if not monthly_hist2.empty:
        monthly_hist2["month_num"] = pd.to_datetime(monthly_hist2["date"] + "-01").dt.month
        seasonality = monthly_hist2.groupby("month_num")["units_sold"].mean().reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        seasonality["month_name"] = seasonality["month_num"].apply(lambda x: month_names[x - 1])

        fig_season = px.line(
            seasonality, x="month_name", y="units_sold",
            title="Average Monthly Seasonality",
            markers=True,
            template="plotly_dark",
        )
        fig_season.update_traces(line_color="#e94560")
        st.plotly_chart(fig_season, use_container_width=True)


# ============================================================
# TAB 3 – BRAND INSIGHTS
# ============================================================
with tab3:
    st.markdown('<div class="section-title">🏆 Brand Performance</div>', unsafe_allow_html=True)

    brand_df = pd.DataFrame(summary.get("top_brands", []))
    if brand_df.empty:
        st.info("No 'brand' column found in the dataset.")
    else:
        fig_brand = px.bar(
            brand_df.head(10), x="brand", y="units_sold",
            title="Top 10 Brands by Total Units Sold",
            color="units_sold",
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig_brand.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_brand, use_container_width=True)

        st.markdown('<div class="section-title">🔮 Per-Brand 2027 Forecast</div>', unsafe_allow_html=True)
        selected_brand = st.selectbox("Select brand to forecast", options=brand_df["brand"].tolist())

        if st.button("Generate Brand Forecast"):
            with st.spinner(f"Forecasting for {selected_brand} …"):
                resp = requests.get(f"{API_BASE}/brand-forecast", params={"brand": selected_brand})
            if resp.status_code == 200:
                bdata = resp.json()
                forecasts = bdata.get("forecasts", {})
                if selected_brand in forecasts:
                    bfc = pd.DataFrame(forecasts[selected_brand])
                    bfc_2027 = bfc[bfc["ds"].str.startswith("2027")]
                    if not bfc_2027.empty:
                        fig_b = go.Figure()
                        # Historical (full prophet fitted)
                        bfc_hist = bfc[~bfc["ds"].str.startswith(("2025", "2026", "2027"))]
                        fig_b.add_trace(go.Scatter(
                            x=bfc_hist["ds"], y=bfc_hist["yhat"].clip(lower=0),
                            name="Historical Fitted",
                            line=dict(color="#4fc3f7"),
                        ))
                        fig_b.add_trace(go.Scatter(
                            x=bfc_2027["ds"], y=bfc_2027["yhat"].clip(lower=0),
                            name="2027 Forecast",
                            line=dict(color="#e94560", dash="dash"),
                            mode="lines+markers",
                        ))
                        fig_b.update_layout(
                            template="plotly_dark",
                            title=f"{selected_brand} — 2027 Monthly Forecast",
                            height=400,
                        )
                        st.plotly_chart(fig_b, use_container_width=True)
                        st.metric(
                            f"Projected 2027 Total Units — {selected_brand}",
                            f"{int(bfc_2027['yhat'].clip(lower=0).sum()):,}"
                        )
            else:
                st.error(resp.json().get("detail", "Error"))


# ============================================================
# TAB 4 – MODEL METRICS
# ============================================================
with tab4:
    st.markdown('<div class="section-title">🤖 Model Performance Metrics</div>', unsafe_allow_html=True)

    metrics = forecast_data.get("metrics", {})

    if metrics:
        col_p, col_x = st.columns(2)

        def render_metrics(name, m, col):
            with col:
                st.markdown(f"#### {name}")
                c1, c2 = st.columns(2)
                c1.metric("MAE", f"{m.get('mae', 0):,.0f}")
                c2.metric("RMSE", f"{m.get('rmse', 0):,.0f}")
                c3, c4 = st.columns(2)
                c3.metric("R²", f"{m.get('r2', 0):.4f}")
                c4.metric("MAPE", f"{m.get('mape', 0):.1f}%")

        if "prophet" in metrics:
            render_metrics("🔵 Prophet", metrics["prophet"], col_p)
        if "xgboost" in metrics:
            render_metrics("🟡 XGBoost", metrics["xgboost"], col_x)

        st.markdown("---")
        st.markdown("**Metric Comparison**")

        model_names = []
        maes, rmses, r2s, mapes = [], [], [], []
        for mname, m in metrics.items():
            model_names.append(mname.capitalize())
            maes.append(m.get("mae", 0))
            rmses.append(m.get("rmse", 0))
            r2s.append(m.get("r2", 0))
            mapes.append(m.get("mape", 0))

        fig_m = make_subplots(rows=1, cols=3, subplot_titles=["MAE", "RMSE", "R²"])
        colors = ["#e94560", "#ffd700"]
        for i, (name, mae, rmse, r2) in enumerate(zip(model_names, maes, rmses, r2s)):
            fig_m.add_trace(go.Bar(name=name, x=[name], y=[mae], marker_color=colors[i], showlegend=False), row=1, col=1)
            fig_m.add_trace(go.Bar(name=name, x=[name], y=[rmse], marker_color=colors[i], showlegend=False), row=1, col=2)
            fig_m.add_trace(go.Bar(name=name, x=[name], y=[r2], marker_color=colors[i], showlegend=True), row=1, col=3)

        fig_m.update_layout(template="plotly_dark", height=350, title="Model Comparison")
        st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **About the models:**
    | Model | Description |
    |-------|-------------|
    | **Prophet** | Facebook's time-series model. Captures yearly seasonality, Indian quarterly cycles, and trend changes automatically. |
    | **XGBoost** | Gradient-boosted tree model trained on lag (1/3/12 month) and rolling-average features. Evaluated with 5-fold time-series CV. |
    """)
