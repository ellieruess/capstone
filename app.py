import json
import os
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import pydeck as pdk
from pathlib import Path
from datetime import date, timedelta
import joblib

alt.data_transformers.disable_max_rows()

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Logistics Analytics", layout="wide")

MAP_HEIGHT = 480  # px height for pydeck maps and their legends

# Optional hint if not launched via `streamlit run`
try:
    import streamlit.runtime as _rt
    if not _rt.exists():
        st.warning("You're running this script outside of Streamlit. Launch with: streamlit run app.py")
except Exception:
    pass

# ---------------------------------
# Warehouse → City map (NO UI OUTPUT)
# ---------------------------------
WAREHOUSE_TO_CITY = {
    "Warehouse_MIA": "Miami",
    "Warehouse_LA": "Los Angeles",
    "Warehouse_BOS": "Boston",
    "Warehouse_SF": "San Francisco",
    "Warehouse_ATL": "Atlanta",
    "Warehouse_CHI": "Chicago",
    "Warehouse_HOU": "Houston",
    "Warehouse_SEA": "Seattle",
    "Warehouse_NYC": "New York City",
    "Warehouse_DEN": "Denver",
}

# ---------------------------------
# City → (lat, lon) map (global, reused)
# ---------------------------------
CITY_COORDS = {
    "atlanta": (33.7490, -84.3880), "chicago": (41.8781, -87.6298), "miami": (25.7617, -80.1918),
    "new york": (40.7128, -74.0060), "los angeles": (34.0522, -118.2437), "san francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321), "boston": (42.3601, -71.0589), "dallas": (32.7767, -96.7970),
    "houston": (29.7604, -95.3698), "austin": (30.2672, -97.7431), "denver": (39.7392, -104.9903),
    "phoenix": (33.4484, -112.0740), "philadelphia": (39.9526, -75.1652), "washington": (38.9072, -77.0369),
    "minneapolis": (44.9778, -93.2650), "detroit": (42.3314, -83.0458), "san diego": (32.7157, -117.1611),
    "orlando": (28.5383, -81.3792), "tampa": (27.9506, -82.4572), "charlotte": (35.2271, -80.8431),
    "nashville": (36.1627, -86.7816), "indianapolis": (39.7684, -86.1581), "columbus": (39.9612, -82.9988),
    "cleveland": (41.4993, -81.6944), "cincinnati": (39.1031, -84.5120), "kansas city": (39.0997, -94.5786),
    "st. louis": (38.6270, -90.1994), "st louis": (38.6270, -90.1994), "new orleans": (29.9511, -90.0715),
    "salt lake city": (40.7608, -111.8910), "las vegas": (36.1699, -115.1398), "portland": (45.5152, -122.6784),
    "san jose": (37.3382, -121.8863), "raleigh": (35.7796, -78.6382), "jacksonville": (30.3322, -81.6557),
    "pittsburgh": (40.4406, -79.9959), "baltimore": (39.2904, -76.6122), "sacramento": (38.5816, -121.4944),
    "milwaukee": (43.0389, -87.9065), "oklahoma city": (35.4676, -97.5164), "albuquerque": (35.0844, -106.6504),
    "omaha": (41.2565, -95.9345), "louisville": (38.2527, -85.7585), "richmond": (37.5407, -77.4360),
    "providence": (41.8240, -71.4128), "buffalo": (42.8864, -78.8784), "hartford": (41.7658, -72.6734),
    "birmingham": (33.5186, -86.8104), "memphis": (35.1495, -90.0490), "tucson": (32.2226, -110.9747),
    "rochester": (43.1566, -77.6088), "des moines": (41.5868, -93.6250), "boise": (43.6150, -116.2023),
    "little rock": (34.7465, -92.2896), "madison": (43.0722, -89.4008), "albany": (42.6526, -73.7562),
    "honolulu": (21.3069, -157.8583), "anchorage": (61.2181, -149.9003), "san antonio": (29.4241, -98.4936),
    "fort worth": (32.7555, -97.3308), "columbia": (34.0007, -81.0348), "greenville": (34.8526, -82.3940),
    "charleston": (32.7765, -79.9311), "newark": (40.7357, -74.1724), "jersey city": (40.7178, -74.0431),
    "stamford": (41.0534, -73.5387), "trenton": (40.2171, -74.7429)
}

# Optional external mapping: city_latlon.csv in project folder with columns [city, lat, lon]
external_city_map = {}
try:
    city_map_path = os.path.join(os.getcwd(), "city_latlon.csv")
    if os.path.exists(city_map_path):
        _cm = pd.read_csv(city_map_path)
        cols_lower = [c.lower() for c in _cm.columns]
        if "city" in cols_lower and "lat" in cols_lower and "lon" in cols_lower:
            _cm.columns = [c.lower() for c in _cm.columns]
            for _, row in _cm.iterrows():
                nm = str(row["city"]).strip().lower()
                if nm and pd.notna(row["lat"]) and pd.notna(row["lon"]):
                    external_city_map[nm] = (float(row["lat"]), float(row["lon"]))
except Exception:
    pass

# ----------------------
# Route helper functions
# ----------------------
def _derive_route(row):
    origin = str(row.get("Origin_Warehouse", ""))
    dest = str(row.get("Destination", ""))
    origin_part = origin.split("_", 1)[1] if "_" in origin else origin
    return f"{origin_part} \u2192 {dest}"

def prepare_route_df(df_in: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    dfp = df_in.copy()
    # Derive Route
    dfp["Route"] = dfp.apply(_derive_route, axis=1)
    # Delay flag
    if "Status" in dfp.columns:
        dfp["__is_delay__"] = dfp["Status"].astype(str).str.lower().eq("delayed")
    else:
        dfp["__is_delay__"] = False
    # Transit days normalization
    td_col = next((c for c in ["Transit_Days", "transit_days", "TransitDays", "transitDays"] if c in dfp.columns), None)
    dfp["__transit_days__"] = pd.to_numeric(dfp[td_col], errors="coerce") if td_col else pd.NA
    return dfp

def render_route_widget(df_show: pd.DataFrame):
    import altair as alt
    import pandas as pd

    report = st.selectbox("Choose a report", ["# of shipments per route", "Routes with the most delays", "Median transit_days per route"], index=0, help="Pick which route report to view", key="route_report_select")

    top_n = st.slider("Top N routes", min_value=5, max_value=50, value=15, step=1, key="route_top_n_slider")

    dfp = prepare_route_df(df_show)

    if report == "# of shipments per route":
        grouped = dfp.groupby("Route", dropna=False).size().reset_index(name="Shipments")
        grouped = grouped.sort_values("Shipments", ascending=False).head(top_n)
        max_val = pd.to_numeric(grouped["Shipments"], errors="coerce").max()
        max_val = int(max_val) if pd.notna(max_val) else 0
        axis_vals = list(range(0, max_val + 1))
        chart = (
            alt.Chart(grouped)
            .mark_bar()
            .encode(
                x=alt.X("Shipments:Q", title="Shipments", axis=alt.Axis(values=axis_vals, format="d")),
                y=alt.Y("Route:N", sort="-x", title="Route"),
                tooltip=[alt.Tooltip("Route:N"), alt.Tooltip("Shipments:Q", format="d")],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    elif report == "Routes with the most delays":
        grouped = (
            dfp.groupby("Route", dropna=False)["__is_delay__"]
            .agg(Delayed_Count="sum", Total="count")
            .reset_index()
        )
        grouped["Delay_Rate"] = (grouped["Delayed_Count"] / grouped["Total"]).where(grouped["Total"] > 0, 0)
        grouped = grouped.sort_values(["Delayed_Count", "Delay_Rate"], ascending=[False, False]).head(top_n)
        max_val = pd.to_numeric(grouped["Delayed_Count"], errors="coerce").max()
        max_val = int(max_val) if pd.notna(max_val) else 0
        axis_vals = list(range(0, max_val + 1))
        chart = (
            alt.Chart(grouped)
            .mark_bar()
            .encode(
                x=alt.X("Delayed_Count:Q", title="Delayed Shipments", axis=alt.Axis(values=axis_vals, format="d")),
                y=alt.Y("Route:N", sort="-x", title="Route"),
                tooltip=[
                    alt.Tooltip("Route:N"),
                    alt.Tooltip("Delayed_Count:Q", format="d"),
                    alt.Tooltip("Delay_Rate:Q", format=".2%"),
                    alt.Tooltip("Total:Q", format="d"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    else:  # Median transit_days per route
        grouped = (
            dfp.groupby("Route", dropna=False)["__transit_days__"]
            .median()
            .reset_index(name="Median_Transit_Days")
        )
        grouped = grouped.sort_values("Median_Transit_Days", ascending=False).head(top_n)
        max_val = pd.to_numeric(grouped["Median_Transit_Days"], errors="coerce").max()
        max_val = int(max_val) if pd.notna(max_val) else 0
        axis_vals = list(range(0, max_val + 1))
        chart = (
            alt.Chart(grouped)
            .mark_bar()
            .encode(
                x=alt.X("Median_Transit_Days:Q", title="Median Transit Days", axis=alt.Axis(values=axis_vals, format="d")),
                y=alt.Y("Route:N", sort="-x", title="Route"),
                tooltip=[alt.Tooltip("Route:N"), alt.Tooltip("Median_Transit_Days:Q", format=".2f")],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

# ----------------------
# Helpers
# ----------------------

@st.cache_data(show_spinner=False)
def load_data(default_path: str | None) -> pd.DataFrame:
    if default_path and os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        # fall back to a common filename in the current folder
        fallback = os.path.join(os.getcwd(), "logistics_shipments_dataset.csv")
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
        else:
            return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    return df


def fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return "-"


def fmt_currency(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return "-"


def fmt_float(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return "-"


def percent(n):
    try:
        return f"{n * 100:.1f}%"
    except Exception:
        return "-"


# ----------------------
# Load data
# ----------------------
DEFAULT_PATH = os.environ.get("LOGISTICS_CSV_PATH", "logistics_shipments_dataset.csv")
df = load_data(DEFAULT_PATH)
if df.empty:
    st.info(
        "Could not find a CSV. Set LOGISTICS_CSV_PATH to your file or place 'logistics_shipments_dataset.csv' in this folder, then rerun.")
    st.stop()

# ----------------------
# Column checks / mapping
# ----------------------
required_cols = {
    "Origin_Warehouse": ["Origin_Warehouse"],
    "Cost": ["Cost", "Shipment_Cost"],
    "Transit_Days": ["Transit_Days"],
    "Status": ["Status"],
    "Weight_kg": ["Weight_kg"],
    "Carrier": ["Carrier"],
}
colmap: dict[str, str] = {}
cols_lower = {c.lower(): c for c in df.columns}
for want, options in required_cols.items():
    found = None
    for opt in options:
        if opt in df.columns:
            found = opt
            break
        key = opt.lower()
        if key in cols_lower:
            found = cols_lower[key]
            break
    if found is None:
        st.error(f"Missing required column similar to '{want}'. Available: {', '.join(df.columns)}")
        st.stop()
    colmap[want] = found

# Working view with normalized names for core columns
work = df.rename(columns={v: k for k, v in colmap.items()})

# ----------------------
# Global carrier color mapping (consistent across charts)
# ----------------------
CARRIER_ORDER = (
    work.get("Carrier")
    .astype(str)
    .str.strip()
    .replace({"": "Unknown"})
    .dropna()
    .unique()
    .tolist()
)
CARRIER_ORDER = sorted(set(CARRIER_ORDER))
# D3 category 10 palette
_CARRIER_BASE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                        "#bcbd22", "#17becf"]
CARRIER_COLOR_RANGE = [_CARRIER_BASE_COLORS[i % len(_CARRIER_BASE_COLORS)] for i in range(len(CARRIER_ORDER))]

# ----------------------
# Destination column detection (for filters & map)
# ----------------------
dest_candidates_all = ["Destination"]
dest_col = next((c for c in dest_candidates_all if c in work.columns), None)

# ----------------------
# Date column detection for page-wide date filter
# ----------------------
date_col_candidates_page = ["Shipment_Date"]
date_col_page = next((c for c in date_col_candidates_page if c in df.columns), None)
if date_col_page is not None:
    _dates_series = pd.to_datetime(df[date_col_page], errors="coerce")
    _min_date = _dates_series.min()
    _max_date = _dates_series.max()
else:
    _dates_series = None
    _min_date = None
    _max_date = None

# ----------------------
# Header
# ----------------------
st.markdown("<h1 style='margin-bottom:0'>Logistics Analytics</h1>", unsafe_allow_html=True)
st.caption(
    "This application is based on a synthetic US Logistics Dataset, which can be found [here](https://www.kaggle.com/datasets/shahriarkabir/us-logistics-performance-dataset)."
)


# ----------------------
# Filters
# ----------------------
with st.container():
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="large")

    # Shipment Date range (if date column exists)
    with c1:
        if date_col_page is not None and _min_date is not None and _max_date is not None:
            default_start = _min_date.date()
            default_end = _max_date.date()
            date_range = st.date_input("Shipment Date", (default_start, default_end), key="date_range")
        else:
            date_range = None
            st.caption("No shipment date column found")

    # Origin Warehouse
    with c2:
        origins = ["All Warehouses"] + sorted(work["Origin_Warehouse"].dropna().unique().tolist())
        origin_choice = st.selectbox("Origin Warehouse", origins, index=0, key="origin_select")

    # Destination (fixed to 'Destination' column only)
    with c3:
        if dest_col:
            dests = ["All Destinations"] + sorted(work[dest_col].dropna().astype(str).str.strip().unique().tolist())
            dest_choice = st.selectbox("Destination", dests, index=0, key="dest_select")
        else:
            dest_choice = "All Destinations"
            st.caption("No Destination column found")

    # Carrier
    with c4:
        carriers = ["All Carriers"] + sorted(work["Carrier"].dropna().unique().tolist())
        carrier_choice = st.selectbox("Carrier", carriers, index=0, key="carrier_select")

# Apply filters
filtered = work.copy()

# Date filter (apply via df index alignment)
if date_col_page is not None and isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    if start_date is not None and end_date is not None:
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        dates_series = pd.to_datetime(df[date_col_page], errors="coerce")
        mask_date = dates_series.between(start_ts, end_ts, inclusive='both')
        filtered = filtered.loc[mask_date].copy()

# Origin filter
if origin_choice != "All Warehouses":
    filtered = filtered[filtered["Origin_Warehouse"] == origin_choice]

# Destination filter
if dest_col and dest_choice != "All Destinations":
    filtered = filtered[dest_col].astype(str).str.strip() == dest_choice
    filtered = work.loc[filtered].copy()

# Carrier filter
if carrier_choice != "All Carriers":
    filtered = filtered[filtered["Carrier"] == carrier_choice]

if filtered.empty:
    st.warning("No rows match the selected filter.")
    st.stop()

# ----------------------
# KPI widgets (5)
# ----------------------
total_shipments = len(filtered)
avg_cost = filtered["Cost"].mean(numeric_only=True)
avg_weight = filtered["Weight_kg"].mean(numeric_only=True)
avg_transit = filtered["Transit_Days"].mean(numeric_only=True)
delivered_mask = filtered["Status"].astype(str).str.strip().str.lower() == "delivered"
pct_delivered = delivered_mask.mean() if len(filtered) else 0.0
total_cost = filtered["Cost"].sum(numeric_only=True)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
with kpi1:
    st.metric("Total Shipments", fmt_int(total_shipments))
with kpi2:
    st.metric("Average Cost", fmt_currency(avg_cost))
with kpi3:
    st.metric("Average Weight (kg)", fmt_float(avg_weight))
with kpi4:
    st.metric("Average Transit Days", fmt_float(avg_transit))
with kpi5:
    st.metric("% Delivered", percent(pct_delivered))
with kpi6:
    st.metric("Total Logistics Expenditure", fmt_currency(total_cost))

# ----------------------
# Median Cost per Kg-Mile (USD)
# ----------------------
st.markdown("### Median Cost per Kg-Mile (USD)")
st.caption("This report isolates cost from package weight & shipping distance to identify seasonal pricing changes.")
date_col_candidates = ["Shipment_Date"]
date_col = next((c for c in date_col_candidates if c in df.columns), None)

if date_col is None:
    st.info("No shipment date column found to compute seasonality.")
else:
    temp = filtered.copy()
    # align with original df index (assumes same row order)
    try:
        temp[date_col] = df.loc[temp.index, date_col]
    except Exception:
        temp[date_col] = df[date_col].iloc[:len(temp)].values

    # ensure numeric fields
    temp["Cost"] = pd.to_numeric(temp["Cost"], errors="coerce")
    temp["Weight_kg"] = pd.to_numeric(temp["Weight_kg"], errors="coerce") if "Weight_kg" in temp.columns else np.nan
    temp["Distance_miles"] = pd.to_numeric(temp["Distance_miles"],
                                           errors="coerce") if "Distance_miles" in temp.columns else np.nan

    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, "Cost", "Weight_kg", "Distance_miles"])

    # avoid division by zero or negative values
    temp = temp[(temp["Weight_kg"] > 0) & (temp["Distance_miles"] > 0)]
    if temp.empty:
        st.info("No rows with valid cost, weight, distance, and dates after filtering.")
    else:
        # metric: cost per kg per mile
        temp["cost_per_kg_mi"] = temp["Cost"] / (temp["Weight_kg"] * temp["Distance_miles"])
        temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=["cost_per_kg_mi"])

        if temp.empty:
            st.info("No rows with finite cost per kg·mile after filtering.")
        else:
            # Decide granularity based on selected date span
            dt_min = temp[date_col].min()
            dt_max = temp[date_col].max()
            span_days = (dt_max - dt_min).days if pd.notna(dt_min) and pd.notna(dt_max) else 9999
            use_week = span_days < 92  # ~3 months

            # Robust y-axis using percentiles
            q_low, q_high = temp["cost_per_kg_mi"].quantile([0.05, 0.95]).tolist()
            if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low >= q_high:
                y_domain = None
            else:
                pad_low = max(0.0, q_low * 0.9)
                pad_high = q_high * 1.1
                y_domain = [pad_low, pad_high]

            rng = np.random.default_rng(42)

            if use_week:
                # ----- Week-to-Week -----
                week_start = temp[date_col].dt.to_period('W-MON').apply(lambda p: p.start_time.normalize())
                weeks_sorted = sorted(week_start.unique())
                week_to_idx = {w: i + 1 for i, w in enumerate(weeks_sorted)}
                temp = temp.assign(
                    WeekStart=week_start,
                    WeekIndex=[week_to_idx[w] for w in week_start],
                )
                temp = temp.assign(WeekJitter=temp["WeekIndex"] + rng.uniform(-0.2, 0.2, len(temp)))
                week_labels = [w.strftime('%b %d') for w in weeks_sorted]
                axis_values = list(range(1, len(week_labels) + 1))
                labels_js = "[" + ",".join([f"'{lbl}'" for lbl in week_labels]) + "]"
                axis = alt.Axis(title="Week (start date)", values=axis_values, labelExpr=f"{labels_js}[datum.value-1]")

                y_enc = alt.Y(
                    "cost_per_kg_mi:Q",
                    title=None,
                    axis=alt.Axis(format="$,.4f"),
                    scale=alt.Scale(domain=y_domain, clamp=True) if y_domain else alt.Scale()
                )

                scatter_bg = (
                    alt.Chart(temp)
                    .mark_circle(size=18, opacity=0.18)
                    .encode(
                        x=alt.X("WeekJitter:Q", axis=axis, title="Week"),
                        y=y_enc,
                        tooltip=[
                            alt.Tooltip("WeekStart:T", title="Week of"),
                            alt.Tooltip("cost_per_kg_mi:Q", title="$ per kg·mi", format="$.4f"),
                            alt.Tooltip("Carrier:N", title="Carrier")
                        ]
                    )
                )

                weekly = (
                    temp.groupby(["WeekIndex"], as_index=False)["cost_per_kg_mi"]
                    .median()
                    .rename(columns={"cost_per_kg_mi": "MedianCPKMi"})
                )
                weekly["WeekLabel"] = [week_labels[i - 1] for i in weekly["WeekIndex"]]

                base = alt.Chart(weekly).encode(
                    x=alt.X("WeekIndex:Q", axis=axis),
                    y=alt.Y("MedianCPKMi:Q", title="Median $ per kg-mile",
                            axis=alt.Axis(format="$,.4f"),
                            scale=alt.Scale(domain=y_domain, clamp=True) if y_domain else alt.Scale()),
                    tooltip=[
                        alt.Tooltip("WeekLabel:N", title="Week"),
                        alt.Tooltip("MedianCPKMi:Q", title="Median $/kg·mi", format="$.4f"),
                    ]
                )
                points = base.mark_circle(size=90, opacity=0.9)
                trend = base.transform_loess("WeekIndex", "MedianCPKMi", bandwidth=0.6).mark_line(size=3)

                st.altair_chart((scatter_bg + points + trend).properties(height=380), use_container_width=True)
            else:
                # ----- Month-to-Month (no gaps; deterministic placement) -----
                # Position each point within its month using day-of-month (not random jitter)
                temp["Month"] = temp[date_col].dt.month
                days_in_month = temp[date_col].dt.days_in_month
                day = temp[date_col].dt.day

                # MonthPos spans [Month-0.5, Month+0.5) so there are no inter-month gaps
                temp["MonthPos"] = temp["Month"] + ((day - 0.5) / days_in_month) - 0.5

                axis = alt.Axis(
                    title="Month",
                    values=list(range(1, 13)),
                    labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value-1]"
                )

                x_scale = alt.Scale(domain=[0.5, 12.5], nice=False, zero=False)

                y_enc = alt.Y(
                    "cost_per_kg_mi:Q",
                    title=None,
                    axis=alt.Axis(format="$,.4f"),
                    scale=alt.Scale(domain=y_domain, clamp=True) if y_domain else alt.Scale()
                )

                scatter_bg = (
                    alt.Chart(temp)
                    .mark_circle(size=18, opacity=0.18)
                    .encode(
                        x=alt.X("MonthPos:Q", axis=axis, title="Month", scale=x_scale),
                        y=y_enc,
                        tooltip=[
                            alt.Tooltip("Month:Q", title="Month", format=".0f"),
                            alt.Tooltip("cost_per_kg_mi:Q", title="$ per kg·mi", format="$.4f"),
                            alt.Tooltip("Carrier:N", title="Carrier")
                        ]
                    )
                )

                monthly = (
                    temp.groupby("Month", as_index=False)["cost_per_kg_mi"]
                    .median()
                    .rename(columns={"cost_per_kg_mi": "MedianCPKMi"})
                )

                base = alt.Chart(monthly).encode(
                    x=alt.X("Month:Q", axis=axis, scale=x_scale),
                    y=alt.Y(
                        "MedianCPKMi:Q",
                        title="Median $ per kg-mile",
                        axis=alt.Axis(format="$,.4f"),
                        scale=alt.Scale(domain=y_domain, clamp=True) if y_domain else alt.Scale()
                    )
                )
                points = base.mark_circle(size=90, opacity=0.9)
                trend = base.transform_loess("Month", "MedianCPKMi", bandwidth=0.6).mark_line(size=3)

                st.altair_chart((scatter_bg + points + trend).properties(height=380), use_container_width=True)
                if y_domain:
                    st.caption(
                        f"Note: Y-axis set to ~5th–95th percentile range ({y_domain[0]:.4f}–{y_domain[1]:.4f} $/kg·mi)."
                    )

# ----------------------
# Transit Days per Mile (Days/mi) — below the Cost per Kg-Mile chart
# ----------------------
st.markdown("### Transit Days per Mile (Days/mi)")
st.caption("This report isolates transit days from shipping distance to identify seasonal trends in delivery time.")
date_col_td = "Shipment_Date" if "Shipment_Date" in df.columns else None
if date_col_td is None:
    st.info("No shipment date column found to compute monthly transit days per mile.")
else:
    tdpm = filtered.copy()

    # align with original df index (same pattern used above)
    try:
        tdpm[date_col_td] = df.loc[tdpm.index, date_col_td]
    except Exception:
        tdpm[date_col_td] = df[date_col_td].iloc[:len(tdpm)].values

    # clean + required cols
    tdpm[date_col_td] = pd.to_datetime(tdpm[date_col_td], errors="coerce")
    tdpm["Transit_Days"] = pd.to_numeric(tdpm["Transit_Days"], errors="coerce")
    tdpm["Distance_miles"] = pd.to_numeric(tdpm.get("Distance_miles"), errors="coerce")

    tdpm = tdpm.dropna(subset=[date_col_td, "Transit_Days", "Distance_miles"])
    tdpm = tdpm[tdpm["Distance_miles"] > 0]

    if tdpm.empty:
        st.info("No rows with valid transit days, distance, and dates after filtering.")
    else:
        # metric: transit days per mile
        tdpm["days_per_mile"] = tdpm["Transit_Days"] / tdpm["Distance_miles"] * 100
        tdpm = tdpm.replace([np.inf, -np.inf], np.nan).dropna(subset=["days_per_mile"])

        if tdpm.empty:
            st.info("No rows with finite days per mile after cleaning.")
        else:
            # --- Month encoding with deterministic placement (no gaps) ---
            tdpm["Month"] = tdpm[date_col_td].dt.month
            days_in_month = tdpm[date_col_td].dt.days_in_month
            day = tdpm[date_col_td].dt.day
            # MonthPos spans [Month-0.5, Month+0.5)
            tdpm["MonthPos"] = tdpm["Month"] + ((day - 0.5) / days_in_month) - 0.5

            # Optional: friendly month names for tooltips
            tdpm["MonthName"] = tdpm[date_col_td].dt.strftime("%b")

            # Robust Y domain (5th–95th percentiles) – keep your existing logic
            ql, qh = tdpm["days_per_mile"].quantile([0.05, 0.95]).tolist()
            y_dom = [max(0.0, ql * 0.9), qh * 1.1] if np.isfinite(ql) and np.isfinite(qh) and ql < qh else None

            month_axis = alt.Axis(
                title="Month",
                values=list(range(1, 13)),
                labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value-1]"
            )
            # Fix the x domain so month bands are contiguous with no outer padding
            x_scale = alt.Scale(domain=[0.5, 12.5], nice=False, zero=False)

            y_enc = alt.Y(
                "days_per_mile:Q",
                title=None,
                axis=alt.Axis(format=".5f"),
                scale=alt.Scale(domain=y_dom, clamp=True) if y_dom else alt.Scale()
            )

            scatter_bg = (
                alt.Chart(tdpm)
                .mark_circle(size=18, opacity=0.18)
                .encode(
                    x=alt.X("MonthPos:Q", axis=month_axis, title="Month", scale=x_scale),
                    y=y_enc,
                    tooltip=[
                        alt.Tooltip("MonthName:N", title="Month"),
                        alt.Tooltip("days_per_mile:Q", title="Days per mile", format=".5f"),
                        alt.Tooltip("Carrier:N", title="Carrier"),
                    ]
                )
            )

            monthly = (
                tdpm.groupby("Month", as_index=False)["days_per_mile"]
                .median()
                .rename(columns={"days_per_mile": "MedianDaysPerMile"})
            )

            points = (
                alt.Chart(monthly)
                .mark_circle(size=90, opacity=0.9)
                .encode(
                    x=alt.X("Month:Q", axis=month_axis, title="Month", scale=x_scale),
                    y=alt.Y(
                        "MedianDaysPerMile:Q",
                        title="Median transit days per mile",
                        axis=alt.Axis(format=".5f"),
                        scale=alt.Scale(domain=y_dom, clamp=True) if y_dom else alt.Scale()
                    ),
                    tooltip=[
                        alt.Tooltip("Month:Q", title="Month", format=".0f"),
                        alt.Tooltip("MedianDaysPerMile:Q", title="Median days/mi", format=".5f"),
                    ]
                )
            )

            trend = (
                alt.Chart(monthly)
                .transform_loess("Month", "MedianDaysPerMile", bandwidth=0.6)
                .mark_line(size=3)
                .encode(
                    x=alt.X("Month:Q", scale=x_scale),  # share same contiguous month domain
                    y=alt.Y("MedianDaysPerMile:Q",
                            scale=alt.Scale(domain=y_dom, clamp=True) if y_dom else alt.Scale())
                )
            )

            st.altair_chart((scatter_bg + points + trend).properties(height=380), use_container_width=True)
            if y_dom:
                st.caption(
                    f"Note: Y-axis set to ~5th–95th percentile range ({y_dom[0]:.5f}–{y_dom[1]:.5f} days/mi)."
                )

# ----------------------
# Route Widget (below chart)
# ----------------------
st.markdown("### Route Insights")
render_route_widget(filtered)

# ----------------------
# Carriers & Package Weights
# ----------------------
st.markdown("### Carriers & Package Weights")
left_c, right_c = st.columns(2, gap="large")

with left_c:
    st.markdown("### Logistics Distribution")
    dist_options = {
        "Transit Days": "Transit_Days",
        "Package Weight": "Weight_kg",
        "Cost": None,  # resolved below
    }
    dist_choice = st.selectbox(
        "Select metric",
        options=list(dist_options.keys()),
        index=0,
        key="logistics_distribution_metric",
    )

    _df = filtered.copy()
    col_name = None
    x_title = ""
    if dist_choice == "Transit Days":
        col_name = "Transit_Days"
        _df[col_name] = pd.to_numeric(_df[col_name], errors="coerce")
        x_title = "Transit Days"
    elif dist_choice == "Package Weight":
        candidates = ["Weight_kg", "Package_Weight", "Weight"]
        col_name = next((c for c in candidates if c in _df.columns), None)
        if col_name is None:
            st.info("No weight column found (expected one of: Weight_kg, Package_Weight, Weight).")
        else:
            _df[col_name] = pd.to_numeric(_df[col_name], errors="coerce")
            x_title = "Package Weight (kg)" if col_name.lower().endswith("kg") or "kg" in col_name.lower() else "Package Weight"
    else:  # Cost
        cost_candidates = ["Cost", "Shipment_Cost", "total_cost", "Total_Cost", "Shipping_Cost"]
        col_name = next((c for c in cost_candidates if c in _df.columns), None)
        if col_name is None:
            st.info("No cost column found (expected one of: Cost, Shipment_Cost, total_cost).")
        else:
            _df[col_name] = pd.to_numeric(_df[col_name], errors="coerce")
            x_title = "Shipment Cost"

    if col_name is not None:
        vals = _df[col_name].dropna()
        if vals.empty:
            st.info("No valid values to plot after cleaning.")
        else:
            cap = float(vals.quantile(0.99))
            eps = 1e-9 * (abs(cap) + 1.0)
            mask_excluded = vals > (cap + eps)
            excluded_count = int(mask_excluded.sum())
            vals_plot = vals.where(~mask_excluded, cap)

            import altair as alt
            bin_config = alt.Bin(step=1.0) if dist_choice == "Transit Days" else alt.Bin(maxbins=40)

            hist = (
                alt.Chart(pd.DataFrame({col_name: vals_plot}))
                .mark_bar()
                .encode(
                    x=alt.X(f"{col_name}:Q", bin=bin_config, title=x_title or col_name.replace("_", " ")),
                    y=alt.Y("count()", title="Number of Shipments"),
                    tooltip=[alt.Tooltip("count()", title="Count")],
                )
                .properties(height=360)
            )
            st.altair_chart(hist, use_container_width=True)
            if excluded_count > 0:
                st.caption(f"Note: {excluded_count} values above {cap:.2f} were capped to avoid skew.")

with right_c:
    st.markdown("### Delivery Performance by Carrier")
    sort_options = {
        "on_time_rate": "On-Time Rate",
        "avg_transit_days": "Average Transit Days",
        "total_shipments": "Total Shipments",
    }
    sort_metric = st.selectbox(
        "Sort by",
        options=list(sort_options.keys()),
        format_func=lambda k: sort_options[k],
        index=0,
        key="perf_sort_metric",
    )
    df2 = filtered.copy()

    date_ship_col = next((c for c in ["Shipment_Date"] if c in df.columns), None)
    date_deliv_col = next((c for c in ["Delivery_Date"] if c in df.columns), None)

    if date_ship_col is None or date_deliv_col is None:
        st.info("Shipment and Delivery date columns are required for delivery performance (not found).")
    else:
        try:
            df2[date_ship_col] = df.loc[df2.index, date_ship_col]
            df2[date_deliv_col] = df.loc[df2.index, date_deliv_col]
        except Exception:
            df2[date_ship_col] = df[date_ship_col].iloc[:len(df2)].values
            df2[date_deliv_col] = df[date_deliv_col].iloc[:len(df2)].values

        df2[date_ship_col] = pd.to_datetime(df2[date_ship_col], errors="coerce")
        df2[date_deliv_col] = pd.to_datetime(df2[date_deliv_col], errors="coerce")

        with_deliv = df2.dropna(subset=[date_ship_col, date_deliv_col]).copy()
        if with_deliv.empty:
            st.info("No shipments with both shipment and delivery dates after filtering.")
        else:
            with_deliv["Actual_Transit_Days"] = (with_deliv[date_deliv_col] - with_deliv[date_ship_col]).dt.days
            with_deliv["On_Time"] = with_deliv["Actual_Transit_Days"] <= pd.to_numeric(with_deliv["Transit_Days"], errors="coerce")

            total_all = df2.groupby("Carrier", dropna=False).size().rename("total_shipments_all").reset_index()

            perf = (
                with_deliv.groupby("Carrier", dropna=False)
                .agg(
                    on_time_rate=("On_Time", lambda s: float(s.mean()) * 100.0),
                    avg_transit_days=("Actual_Transit_Days", "mean"),
                    total_with_delivery=("On_Time", "count")
                )
                .reset_index()
            )

            perf = perf.merge(total_all, on="Carrier", how="outer")
            perf = perf.fillna(
                {"on_time_rate": 0.0, "avg_transit_days": 0.0, "total_with_delivery": 0, "total_shipments_all": 0})
            perf["Carrier"] = perf["Carrier"].astype(str).str.strip().replace({"": "Unknown"})
            perf["avg_transit_days"] = perf["avg_transit_days"].round(2)
            perf["on_time_rate"] = perf["on_time_rate"].round(1)

            if sort_metric == "on_time_rate":
                perf = perf.sort_values("on_time_rate", ascending=False)
                y_field = "on_time_rate"; y_title = "On-Time Rate (%)"
            elif sort_metric == "avg_transit_days":
                perf = perf.sort_values("avg_transit_days", ascending=True)
                y_field = "avg_transit_days"; y_title = "Average Transit Days"
            else:
                perf = perf.sort_values("total_shipments_all", ascending=False)
                y_field = "total_shipments_all"; y_title = "Total Shipments"

            carrier_order = perf["Carrier"].astype(str).tolist()

            chart = (
                alt.Chart(perf)
                .mark_bar()
                .encode(
                    x=alt.X("Carrier:N", sort=carrier_order, title="Carrier"),
                    y=alt.Y(f"{y_field}:Q", title=y_title),
                    color=alt.Color("Carrier:N", scale=alt.Scale(domain=CARRIER_ORDER, range=CARRIER_COLOR_RANGE), legend=None),
                    tooltip=[
                        alt.Tooltip("Carrier:N", title="Carrier"),
                        alt.Tooltip("on_time_rate:Q", title="On-Time Rate (%)", format=".1f"),
                        alt.Tooltip("avg_transit_days:Q", title="Avg Transit Days", format=".2f"),
                        alt.Tooltip("total_with_delivery:Q", title="Shipments with Delivery"),
                        alt.Tooltip("total_shipments_all:Q", title="Total Shipments")
                    ]
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)

# ----------------------
# Origin Warehouses Map
# ----------------------
st.markdown("### Origin Warehouses")

orig_map_df = filtered.copy()

if "Origin_Warehouse" not in orig_map_df.columns:
    st.info("No Origin_Warehouse column found for origin map.")
else:
    # Convert codes -> city -> (lat, lon) using global maps
    orig_map_df["OriginCity"] = orig_map_df["Origin_Warehouse"].map(WAREHOUSE_TO_CITY)
    orig_map_df = orig_map_df.dropna(subset=["OriginCity"]).copy()

    def _norm_city_for_origin(s: str) -> str:
        s = (s or "").strip().lower()
        s = s.split(",")[0].strip()
        if s == "new york city":
            return "new york"
        return s

    orig_map_df["_city_key"] = orig_map_df["OriginCity"].astype(str).map(_norm_city_for_origin)
    orig_map_df["_lat"] = orig_map_df["_city_key"].map(lambda k: (CITY_COORDS.get(k) or external_city_map.get(k) or (np.nan, np.nan))[0])
    orig_map_df["_lon"] = orig_map_df["_city_key"].map(lambda k: (CITY_COORDS.get(k) or external_city_map.get(k) or (np.nan, np.nan))[1])

    orig_map_df = orig_map_df.dropna(subset=["_lat", "_lon"]).copy()

    if orig_map_df.empty:
        st.info("No rows with valid origin locations after filtering.")
    else:
        grp_o = (
            orig_map_df.groupby(["Origin_Warehouse", "OriginCity", "_lat", "_lon"], dropna=False)
            .agg(shipments=("Carrier", "count"))
            .reset_index()
            .rename(columns={"_lat": "lat", "_lon": "lon"})
        )

        if grp_o.empty:
            st.info("No origin warehouses to map.")
        else:
            grp_o["radius_px"] = (grp_o["shipments"].astype(float) / 10.0)

            uniq_codes = grp_o["Origin_Warehouse"].astype(str).unique().tolist()
            palette = [
                [31, 119, 180, 200], [255, 127, 14, 200], [44, 160, 44, 200], [214, 39, 40, 200], [148, 103, 189, 200],
                [140, 86, 75, 200], [227, 119, 194, 200], [127, 127, 127, 200], [188, 189, 34, 200], [23, 190, 207, 200]
            ]
            code_to_color = {code: palette[i % len(palette)] for i, code in enumerate(uniq_codes)}
            grp_o["color"] = grp_o["Origin_Warehouse"].astype(str).map(code_to_color)

            center_lat = float(grp_o["lat"].mean())
            center_lon = float(grp_o["lon"].mean())

            origin_layer = pdk.Layer(
                "ScatterplotLayer",
                data=grp_o,
                get_position="[lon, lat]",
                get_radius="radius_px",
                radius_units="pixels",
                get_fill_color="color",
                get_line_color="color",
                stroked=True,
                line_width_min_pixels=1.5,
                radius_scale=1,
                pickable=True,
                auto_highlight=True,
                opacity=0.65,
            )

            origin_tooltip = {"text": "Warehouse: {Origin_Warehouse}\nCity: {OriginCity}\nShipments: {shipments}"}

            origin_deck = pdk.Deck(
                layers=[origin_layer],
                initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3),
                tooltip=origin_tooltip
            )
            left_o, right_o = st.columns([5, 2], gap="large")
            with left_o:
                st.pydeck_chart(origin_deck, height=MAP_HEIGHT)
            try:
                _legend_o = grp_o.groupby("Origin_Warehouse", dropna=False)["shipments"].sum().reset_index()
                _legend_o = _legend_o.sort_values("shipments", ascending=False)
                _order_o = _legend_o["Origin_Warehouse"].astype(str).tolist()

                items_o = []
                for code_lbl in _order_o:
                    rgba = code_to_color.get(code_lbl, [127, 127, 127, 200])
                    r, g, b, a = rgba
                    a_css = round(a / 255.0, 3)
                    swatch = f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background-color: rgba({r},{g},{b},{a_css});border:1px solid #888;margin-right:8px;"></span>'
                    row = grp_o[grp_o["Origin_Warehouse"].astype(str) == code_lbl]
                    city_lbl = row["OriginCity"].iloc[0] if len(row) else ""
                    items_o.append(
                        f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:12px;">{swatch}<span>{code_lbl}'
                        + (f' &mdash; {city_lbl}' if city_lbl else '') + '</span></div>')

                legend_o_html = f'<div style="display:flex;flex-direction:column;row-gap:6px;max-height:{MAP_HEIGHT}px;height:{MAP_HEIGHT}px;overflow:auto;margin:6px 0 10px 0;">' + ''.join(items_o) + '</div>'
                with right_o:
                    st.markdown(legend_o_html, unsafe_allow_html=True)
            except Exception:
                pass

# ----------------------
# Destination Map
# ----------------------
st.markdown("### Destination Map")

map_df = filtered.copy()

def normalize_city(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    base = s.split(",")[0].strip().lower()
    return base

dest_candidates_local = ["Destination"]
dest_col_for_map = next((c for c in dest_candidates_local if c in map_df.columns), None)
if dest_col_for_map:
    cities = map_df[dest_col_for_map].astype(str).map(normalize_city)
    lats, lons = [], []
    hit, miss = 0, 0
    for c in cities:
        coord = external_city_map.get(c) or CITY_COORDS.get(c)
        if coord is None:
            lats.append(None); lons.append(None); miss += 1
        else:
            lat, lon = coord
            lats.append(lat); lons.append(lon); hit += 1
    map_df["_lat"] = lats
    map_df["_lon"] = lons
    lat_col, lon_col = "_lat", "_lon"
else:
    st.caption("No Destination column found for mapping city names.")
    lat_col, lon_col = None, None

if lat_col and lon_col:
    map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors="coerce")
    map_df[lon_col] = pd.to_numeric(map_df[lon_col], errors="coerce")
    map_df = map_df.dropna(subset=[lat_col, lon_col])

    if map_df.empty:
        st.info("No rows with valid destination coordinates after filtering.")
    else:
        dest_col_for_label = "Destination" if "Destination" in map_df.columns else None
        if dest_col_for_label:
            map_df["DestLabel"] = map_df[dest_col_for_label].astype(str).str.strip()
        else:
            map_df["DestLabel"] = ""

        if map_df["DestLabel"].astype(bool).any():
            grp = map_df.groupby("DestLabel", dropna=False).agg(
                shipments=("Carrier", "count"),
                lat=(lat_col, "mean"),
                lon=(lon_col, "mean")
            ).reset_index()
        else:
            tmp = map_df.copy()
            tmp["_lat_r"] = tmp[lat_col].round(3)
            tmp["_lon_r"] = tmp[lon_col].round(3)
            grp = tmp.groupby(["_lat_r", "_lon_r"], dropna=False).agg(
                shipments=("Carrier", "count"),
                lat=(lat_col, "mean"),
                lon=(lon_col, "mean")
            ).reset_index()
            grp["DestLabel"] = ""

        if grp.empty:
            st.info("No destinations to map after grouping.")
        else:
            grp["radius_px"] = (grp["shipments"].astype(float) / 10.0)

            if grp["DestLabel"].astype(bool).any():
                uniq = sorted(grp["DestLabel"].astype(str).unique().tolist())
            else:
                uniq = [""]

            palette = [
                [31, 119, 180, 180], [255, 127, 14, 180], [44, 160, 44, 180], [214, 39, 40, 180], [148, 103, 189, 180],
                [140, 86, 75, 180], [227, 119, 194, 180], [127, 127, 127, 180], [188, 189, 34, 180], [23, 190, 207, 180]
            ]
            color_map = {label: palette[i % len(palette)] for i, label in enumerate(uniq)}
            grp["color"] = grp["DestLabel"].astype(str).map(color_map)

            center_lat = float(grp["lat"].mean())
            center_lon = float(grp["lon"].mean())

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=grp,
                get_position="[lon, lat]",
                get_radius="radius_px",
                radius_units="pixels",
                get_fill_color="color",
                get_line_color="color",
                stroked=True,
                line_width_min_pixels=1.5,
                radius_scale=1,
                pickable=True,
                auto_highlight=True,
                opacity=0.65,
            )

            tooltip = {"text": "{DestLabel}\nShipments: {shipments}"} if grp["DestLabel"].astype(bool).any() else {"text": "Shipments: {shipments}"}

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3),
                tooltip=tooltip
            )
            left_d, right_d = st.columns([5, 2], gap="large")
            with left_d:
                st.pydeck_chart(deck, height=MAP_HEIGHT)

            try:
                _legend_df = grp.groupby("DestLabel", dropna=False)["shipments"].sum().reset_index()
                _legend_df = _legend_df.sort_values("shipments", ascending=False)
                _order = _legend_df["DestLabel"].astype(str).tolist()
                _topN = min(15, len(_order))
                _shown = _order[:_topN]
                _remaining = max(0, len(_order) - _topN)

                items = []
                for lbl in _shown:
                    rgba = color_map.get(lbl, [127, 127, 127, 180])
                    r, g, b, a = rgba
                    a_css = round(a / 255.0, 3)
                    swatch = f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background-color: rgba({r},{g},{b},{a_css});border:1px solid #888;margin-right:8px;"></span>'
                    items.append(f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:12px;">{swatch}<span>{lbl}</span></div>')

                extra = f'<div style="font-size:12px;color:#666;margin-top:6px;">+{_remaining} more…</div>' if _remaining > 0 else ''
                legend_html = f'<div style="display:flex;flex-direction:column;row-gap:6px;max-height:{MAP_HEIGHT}px;height:{MAP_HEIGHT}px;overflow:auto;margin:6px 0 10px 0;">' + ''.join(items) + '</div>' + extra
                with right_d:
                    st.markdown(legend_html, unsafe_allow_html=True)
            except Exception:
                pass

# --- Pickle compatibility shim for custom transformer used in preprocessor.pkl ---
try:
    import sys
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_extraction import FeatureHasher

    class RouteHashingEncoder(BaseEstimator, TransformerMixin):
        """Matches the class used during training so joblib can unpickle the preprocessor."""
        def __init__(self, n_features=256):
            self.n_features = int(n_features)
            self.hasher = FeatureHasher(n_features=self.n_features, input_type="string")

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                strings = X.iloc[:, 0].astype(str).tolist()
            else:
                strings = pd.Series(np.asarray(X).ravel()).astype(str).tolist()
            samples = [[s] for s in strings]
            return self.hasher.transform(samples).toarray()

    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "RouteHashingEncoder"):
        setattr(main_mod, "RouteHashingEncoder", RouteHashingEncoder)
except Exception:
    pass

# ======================================================================
# 💡 Shipment Price Prediction Sidebar — Compare All Carriers
# ======================================================================
_PRED_ARTIFACTS_DIR = Path(os.getenv("LOGISTICS_ARTIFACTS_DIR", "model_artifacts"))

try:
    files = sorted(os.listdir(_PRED_ARTIFACTS_DIR))
except Exception as e:
    st.warning(f"Could not list artifacts dir: {e}")

@st.cache_resource(show_spinner=False)
def _pred_load_artifacts():
    """Load pre-trained artifacts."""
    preproc_path = _PRED_ARTIFACTS_DIR / "preprocessor.pkl"
    cost_model_path = _PRED_ARTIFACTS_DIR / "cost_model.pkl"

    preprocessor = joblib.load(preproc_path)
    cost_model = joblib.load(cost_model_path)

    best_params_json = _PRED_ARTIFACTS_DIR / "best_params.json"
    grouped_models_path = _PRED_ARTIFACTS_DIR / "cost_models_grouped.pkl"

    features = None
    cost_calibration = {}
    try:
        with open(best_params_json, "r") as f:
            meta = json.load(f)
        features = meta.get("features")
        cost_calibration = (meta.get("calibration") or {}).get("cost", {}) or {}
    except Exception:
        pass

    grouped_models = None
    if grouped_models_path.exists():
        try:
            grouped_models = joblib.load(grouped_models_path)
        except Exception:
            grouped_models = None

    return {
        "ok": True,
        "preprocessor": preprocessor,
        "cost_model": cost_model,
        "features": features,
        "cost_calibration": cost_calibration,
        "grouped_models": grouped_models,
    }

# --------- feature engineering (mirror of training) ----------
def _pred_feature_engineering(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    for c in ["Carrier", "origin_warehouse", "Destination", "Distance_miles", "Weight_kg", "shipment_date"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["Distance_miles"] = pd.to_numeric(df["Distance_miles"], errors="coerce")
    df["Weight_kg"] = pd.to_numeric(df["Weight_kg"], errors="coerce")
    df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")

    df["month"] = df["shipment_date"].dt.month.fillna(0).astype(int)
    df["day_of_week"] = df["shipment_date"].dt.dayofweek.fillna(0).astype(int)
    df["day_of_month"] = df["shipment_date"].dt.day.fillna(0).astype(int)
    df["is_weekend"] = (df["shipment_date"].dt.dayofweek >= 5).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    df["route"] = df["origin_warehouse"].astype(str) + "→" + df["Destination"].astype(str)

    cstr = df["Carrier"].astype(str)
    df["is_dhl"] = (cstr == "DHL").astype(int)
    df["is_usps"] = (cstr == "USPS").astype(int)
    df["is_ups"] = (cstr == "UPS").astype(int)
    df["is_fedex"] = (cstr == "FedEx").astype(int)

    df["dhl_distance"] = df["is_dhl"] * df["Distance_miles"]
    df["usps_distance"] = df["is_usps"] * df["Distance_miles"]
    df["dhl_weight"] = df["is_dhl"] * df["Weight_kg"]
    df["usps_weight"] = df["is_usps"] * df["Weight_kg"]

    df["carrier_route_count"] = 1

    def _cut(series, bins):
        try:
            return pd.cut(series, bins=bins, labels=False).astype("float").fillna(0).astype(int)
        except Exception:
            return pd.Series([0] * len(series), index=series.index)

    dhl_mask = cstr == "DHL"
    df["dhl_distance_tier"] = 0
    df.loc[dhl_mask, "dhl_distance_tier"] = _cut(df.loc[dhl_mask, "Distance_miles"], [0, 500, 1000, 1500, 2000, 2500, 3000])
    df["dhl_weight_tier"] = 0
    df.loc[dhl_mask, "dhl_weight_tier"] = _cut(df.loc[dhl_mask, "Weight_kg"], [0, 10, 20, 30, 40, 50, 100])
    df["dhl_route_complexity"] = 0
    df["dhl_weekend"] = ((dhl_mask) & (df["is_weekend"] == 1)).astype(int)
    df["dhl_holiday"] = ((dhl_mask) & (df["is_holiday_season"] == 1)).astype(int)

    df["dhl_distance_tier_detailed"] = 0
    df.loc[dhl_mask, "dhl_distance_tier_detailed"] = _cut(
        df.loc[dhl_mask, "Distance_miles"],
        [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 3000]
    )
    df["dhl_very_long_distance"] = ((dhl_mask) & (df["Distance_miles"] > 2000)).astype(int)
    df["dhl_distance_squared"] = df["is_dhl"] * (df["Distance_miles"] ** 2)

    df["dhl_weight_tier_detailed"] = 0
    df.loc[dhl_mask, "dhl_weight_tier_detailed"] = _cut(
        df.loc[dhl_mask, "Weight_kg"],
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 100]
    )
    df["dhl_heavy_package"] = ((dhl_mask) & (df["Weight_kg"] > 30)).astype(int)
    df["dhl_very_heavy_package"] = ((dhl_mask) & (df["Weight_kg"] > 35)).astype(int)
    df["dhl_weight_distance"] = df["is_dhl"] * df["Weight_kg"] * df["Distance_miles"]

    problem_routes = [
        "Warehouse_CHI→Houston",
        "Warehouse_HOU→Chicago",
        "Warehouse_MIA→San Francisco",
        "Warehouse_ATL→Denver",
        "Warehouse_NYC→Phoenix",
    ]
    df["is_problem_route"] = df["route"].isin(problem_routes).astype(int)
    df["is_long_distance_dhl"] = ((dhl_mask) & (df["Distance_miles"] > 1900)).astype(int)
    df["dhl_long_distance"] = df["is_dhl"] * (df["Distance_miles"] > 1900).astype(int)
    df["problem_route_distance"] = df["is_problem_route"] * df["Distance_miles"]

    for c in ["carrier_avg_days", "route_avg_days", "carrier_route_avg_days"]:
        df[c] = 0.0

    return df

def _pred_apply_calibration_and_rules(cost_array: np.ndarray, df_row: pd.DataFrame, cost_calibration: dict) -> np.ndarray:
    out = cost_array.astype(float).copy()

    if isinstance(cost_calibration, dict):
        carrier = str(df_row["Carrier"].iloc[0])
        factors = cost_calibration.get(carrier)
        if isinstance(factors, dict):
            mult = float(factors.get("multiplier", 1.0))
            if np.isfinite(mult) and mult > 0:
                out *= mult

    if str(df_row["Carrier"].iloc[0]) == "DHL":
        route_corrections = {
            "Warehouse_CHI→Houston": 0.7,
            "Warehouse_HOU→Chicago": 0.8,
            "Warehouse_MIA→San Francisco": 0.85,
            "Warehouse_ATL→Denver": 0.9,
            "Warehouse_NYC→Phoenix": 0.9,
        }
        route = str(df_row["route"].iloc[0])
        if route in route_corrections:
            out *= route_corrections[route]

        try:
            if float(df_row["Distance_miles"].iloc[0]) > 1900:
                out *= 0.85
        except Exception:
            pass

    out = np.maximum(out, 0.0)
    return out

# --------- Transit-days estimator (range + delivery window) ----------
def _pick_transit_series(base_df: pd.DataFrame, carrier: str, origin_wh: str, dest: str) -> pd.Series:
    """Return the most specific historical series of Transit_Days we can find."""
    df0 = base_df.copy()
    df0["Transit_Days"] = pd.to_numeric(df0["Transit_Days"], errors="coerce")
    df0 = df0.dropna(subset=["Transit_Days"])

    def _filt(use_carrier=False, use_origin=False, use_dest=False):
        m = pd.Series(True, index=df0.index)
        if use_carrier:
            m &= df0["Carrier"].astype(str) == str(carrier)
        if use_origin:
            m &= df0["Origin_Warehouse"].astype(str) == str(origin_wh)
        if use_dest:
            m &= df0["Destination"].astype(str).str.strip() == str(dest)
        return df0.loc[m, "Transit_Days"]

    # Preference order (most specific → least)
    for use in [
        (True, True, True),
        (True, True, False),
        (False, True, True),
        (True, False, False),
        (False, False, False),
    ]:
        s = _filt(*use)
        if len(s) >= 6:
            return s

        if len(s) >= 1 and '_fallback' not in locals():
            _fallback = s
    return locals().get('_fallback', pd.Series(dtype=float))

def _estimate_transit_range(base_df: pd.DataFrame, carrier: str, origin_wh: str, dest: str, distance_miles: float | None) -> tuple[int, int]:
    s = _pick_transit_series(base_df, carrier, origin_wh, dest)
    low: int; high: int
    if len(s) >= 6:
        q25, q75 = np.quantile(s, [0.25, 0.75])
        low = int(np.floor(q25))
        high = int(np.ceil(q75))
    elif len(s) >= 1:
        med = float(np.median(s))
        low = int(np.floor(med - 1))
        high = int(np.ceil(med + 1))
    else:
        # Distance-based heuristic fallback
        d = float(distance_miles or 0)
        if d < 250:
            low, high = 1, 2
        elif d < 600:
            low, high = 2, 4
        elif d < 1200:
            low, high = 3, 6
        elif d < 2000:
            low, high = 4, 7
        else:
            low, high = 5, 9

    # Business rule: never less than 1 day; ensure high >= low
    low = max(1, low)
    high = max(low, high)
    return low, high

# ---------------------- Sidebar UI ----------------------
with st.sidebar:
    st.markdown("### Predict Shipment Price")
    st.caption("This tool predicts shipment price and package arrival date.")

    _art = _pred_load_artifacts()
    if not _art.get("ok", False):
        st.error(_art.get("err", "Could not load model artifacts."))
    else:
        try:
            _whs = sorted(set(work["Origin_Warehouse"].dropna().astype(str).tolist()))
        except Exception:
            _whs = []
        try:
            _dests = sorted(set(work["Destination"].dropna().astype(str).tolist())) if "Destination" in work.columns else []
        except Exception:
            _dests = []

        # ---- helpers: mapping & distance (no city names shown to user) ----
        from math import radians, sin, cos, asin, sqrt

        def _normalize_city_key(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return ""
            base = s.split(",")[0].strip().lower()
            if base == "new york city":
                base = "new york"
            return base

        def _city_to_coords(city_name: str):
            key = _normalize_city_key(city_name)
            coord = external_city_map.get(key) or CITY_COORDS.get(key)
            if coord is None:
                return None, None
            lat, lon = coord
            return float(lat), float(lon)

        def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
            R = 3958.7613  # miles
            φ1, φ2 = radians(lat1), radians(lat2)
            dφ = radians(lat2 - lat1)
            dλ = radians(lon2 - lon1)
            a = sin(dφ/2)**2 + cos(φ1) * cos(φ2) * sin(dλ/2)**2
            c = 2 * asin(sqrt(a))
            return R * c

        def _warehouse_code_to_city(wh_code: str) -> str | None:
            return WAREHOUSE_TO_CITY.get(str(wh_code), None)

        with st.form("pred_form", border=True):

            top1, top2 = st.columns(2)
            with top1:
                in_origin = st.selectbox("Origin Warehouse", options=_whs or ["Warehouse_CHI"], index=0)
            with top2:
                in_dest = st.selectbox("Destination", options=_dests or ["Houston"], index=0)

            bot1, bot2 = st.columns(2)
            with bot1:
                in_weight = st.number_input("Weight (kg)", min_value=0.0, value=5.0, step=0.5)
            with bot2:
                in_date = st.date_input("Shipment Date", value=date.today())

            submitted = st.form_submit_button("Predict", use_container_width=True)

        if submitted:
            try:
                origin_city = _warehouse_code_to_city(in_origin)
                if not origin_city:
                    st.error(f"Could not map origin warehouse '{in_origin}' to a city. Update WAREHOUSE_TO_CITY.")
                    st.stop()

                o_lat, o_lon = _city_to_coords(origin_city)
                d_lat, d_lon = _city_to_coords(in_dest)

                missing_bits = []
                if o_lat is None or o_lon is None:
                    missing_bits.append("origin warehouse location")
                if d_lat is None or d_lon is None:
                    missing_bits.append(f"destination '{in_dest}'")
                if missing_bits:
                    st.error("Could not find coordinates for " + ", ".join(missing_bits) +
                             ". Add them to CITY_COORDS or city_latlon.csv.")
                    st.stop()

                in_distance = _haversine_miles(o_lat, o_lon, d_lat, d_lon)

                try:
                    carriers_to_score = sorted(set(work["Carrier"].dropna().astype(str).tolist()))
                except Exception:
                    carriers_to_score = []
                if not carriers_to_score:
                    carriers_to_score = ["DHL", "UPS", "USPS", "FedEx"]

                results = []
                for carrier in carriers_to_score:
                    # Estimate transit-days range **before** prediction (for display + delivery window)
                    t_low, t_high = _estimate_transit_range(work, carrier, in_origin, in_dest, in_distance)
                    deliv_start = pd.to_datetime(in_date) + timedelta(days=int(t_low))
                    deliv_end = pd.to_datetime(in_date) + timedelta(days=int(t_high))

                    row = pd.DataFrame([{
                        "Carrier": str(carrier),
                        "origin_warehouse": str(in_origin),
                        "Destination": str(in_dest),
                        "Distance_miles": float(in_distance),
                        "Weight_kg": float(in_weight),
                        "shipment_date": pd.to_datetime(in_date),
                    }])

                    row_fe = _pred_feature_engineering(row)

                    feats = _art.get("features")
                    if isinstance(feats, list) and len(feats) > 0:
                        for miss in feats:
                            if miss not in row_fe.columns:
                                row_fe[miss] = 0 if miss not in ["Carrier", "origin_warehouse", "Destination", "route"] else "Unknown"
                        X_row = row_fe[feats]
                    else:
                        X_row = row_fe.copy()

                    X_pre = _art["preprocessor"].transform(X_row)

                    grouped = _art.get("grouped_models")
                    mdl = grouped.get(carrier) if isinstance(grouped, dict) and carrier in grouped else _art["cost_model"]

                    y_log = mdl.predict(X_pre, num_iteration=getattr(mdl, "best_iteration", None))
                    y_cost = np.expm1(y_log).reshape(-1)
                    y_cost_adj = _pred_apply_calibration_and_rules(y_cost, row_fe, _art.get("cost_calibration", {}))

                    results.append({
                        "Carrier": carrier,
                        "Predicted_Cost": float(y_cost_adj[0]),
                        "Transit_Low": int(t_low),
                        "Transit_High": int(t_high),
                        "Delivery_Start": deliv_start,
                        "Delivery_End": deliv_end
                    })

                # Build results table
                res_df = pd.DataFrame(results).sort_values("Predicted_Cost", ascending=True)
                res_df["Price ($)"] = res_df["Predicted_Cost"].map(lambda v: f"${v:,.2f}")
                res_df["Delivery (est)"] = res_df.apply(
                    lambda r: f"{pd.to_datetime(r['Delivery_Start']).strftime('%b %d')}–{pd.to_datetime(r['Delivery_End']).strftime('%b %d')}",
                    axis=1
                )

                st.markdown("#### Results")
                st.caption(f"Distance (computed): **{in_distance:,.2f} miles**  •  Route: **{in_origin} → {in_dest}**")

                # Chart (with extra tooltips)
                try:
                    import altair as alt
                    plot_df = res_df.copy()
                    plot_df["Delivery_window"] = plot_df["Delivery (est)"]
                    order = plot_df.sort_values("Predicted_Cost")["Carrier"].tolist()
                    chart = (
                        alt.Chart(plot_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Carrier:N", sort=order, title="Carrier"),
                            y=alt.Y("Predicted_Cost:Q", title="Price ($)", axis=alt.Axis(format="$,.2f")),
                            color=alt.Color("Carrier:N", scale=alt.Scale(domain=CARRIER_ORDER, range=CARRIER_COLOR_RANGE), legend=None),
                            tooltip=[
                                alt.Tooltip("Carrier:N"),
                                alt.Tooltip("Predicted_Cost:Q", title="Price ($)", format="$,.2f"),
                                alt.Tooltip("Delivery_window:N", title="Delivery (est)")
                            ]
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    pass

                # Sidebar table: Price + Delivery
                display_cols = ["Carrier", "Price ($)", "Delivery (est)"]
                st.dataframe(res_df[display_cols], use_container_width=True, hide_index=True)

                # Callout for the cheapest option (include window)
                cheapest_row = res_df.iloc[0]
                st.success(
                    f"Lowest predicted price: **{cheapest_row['Carrier']} — ${cheapest_row['Predicted_Cost']:,.2f}**  "
                    f"• Transit **{int(cheapest_row['Transit_Low'])}–{int(cheapest_row['Transit_High'])} days**  "
                    f"• Est **{pd.to_datetime(cheapest_row['Delivery_Start']).strftime('%b %d')}–{pd.to_datetime(cheapest_row['Delivery_End']).strftime('%b %d')}**"
                )

                st.caption("Transit days are estimated from historical data (falling back to distance heuristics) and clamped to a minimum of 1 day.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ----------------------
# Detail table
# ----------------------
with st.expander("Show filtered rows", expanded=False):
    st.dataframe(filtered, width='stretch', hide_index=True)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("Western Governors University - BSCS Capstone - Student ID: 000375111")
