"""
╔══════════════════════════════════════════════════════════════════╗
║                 FRONTEND — dashboard.py                          ║
║   Run AFTER train.py has finished                                ║
║   Command:  streamlit run dashboard.py                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, pickle, warnings, datetime, io
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Chronic Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────
# COLOUR PALETTE
# ───────────────────────────────────────────────
BG0    = "#07071a"
BG1    = "#0d0d20"
BG2    = "#12122c"
BG3    = "#1a1a38"
BG4    = "#22224a"
ACCENT = "#6c63ff"
CYAN   = "#22d3ee"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
RED    = "#f43f5e"
PINK   = "#e879f9"
TXT0   = "#f0eeff"
TXT1   = "#a09cc8"
TXT2   = "#5a5888"
PAL    = [ACCENT, CYAN, GREEN, AMBER, RED, PINK, "#38bdf8", "#fb923c"]

# ───────────────────────────────────────────────
# CSS
# ───────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, html, body, [class*="css"] {{
  font-family: 'Outfit', sans-serif !important;
  box-sizing: border-box;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.6rem 2rem 3rem !important; max-width: 100% !important; }}
[data-testid="stAppViewContainer"] {{ background: {BG0}; }}

::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG1}; }}
::-webkit-scrollbar-thumb {{ background: {BG4}; border-radius: 3px; }}

[data-testid="stSidebar"] {{
  background: {BG1} !important;
  border-right: 1px solid {BG3} !important;
}}
[data-testid="stSidebar"] * {{ color: {TXT0} !important; }}
[data-testid="stSidebar"] label {{
  font-size: 10px !important; text-transform: uppercase !important;
  letter-spacing: 0.10em !important; color: {TXT2} !important; font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stDateInput input {{
  background: {BG2} !important; border: 1px solid {BG3} !important;
  color: {TXT0} !important; border-radius: 8px !important; font-size: 13px !important;
}}
hr {{ border-color: {BG3} !important; }}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0; background: {BG1};
  border-bottom: 1px solid {BG3}; padding: 0 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
  font-size: 11px !important; font-weight: 600 !important;
  padding: 11px 20px !important; color: {TXT2} !important;
  border: none !important; background: transparent !important;
  letter-spacing: 0.06em; text-transform: uppercase;
}}
.stTabs [aria-selected="true"] {{
  color: {ACCENT} !important;
  border-bottom: 2px solid {ACCENT} !important;
  background: rgba(108,99,255,0.06) !important;
}}

.stButton > button {{
  background: linear-gradient(135deg, {ACCENT}, #9b59ff) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-size: 12px !important; font-weight: 700 !important;
  height: 44px !important; width: 100% !important;
  letter-spacing: 0.08em; text-transform: uppercase;
  box-shadow: 0 4px 20px rgba(108,99,255,0.35) !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

[data-testid="stMetric"] {{
  background: {BG2}; border: 1px solid {BG3};
  border-radius: 12px; padding: 14px 18px;
}}
[data-testid="stMetricLabel"] p {{
  font-size: 10px !important; text-transform: uppercase;
  letter-spacing: 0.08em; color: {TXT2} !important;
}}
[data-testid="stMetricValue"] {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 21px !important; color: {TXT0} !important;
}}

.kpi-strip {{
  display: grid; grid-template-columns: repeat(5, 1fr);
  gap: 12px; margin-bottom: 1.6rem;
}}
.kpi-card {{
  background: {BG2}; border: 1px solid {BG3};
  border-radius: 14px; padding: 16px 18px;
  position: relative; overflow: hidden;
}}
.kpi-a::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,{ACCENT},{CYAN}); }}
.kpi-b::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,{CYAN},{GREEN}); }}
.kpi-c::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,{AMBER},{RED}); }}
.kpi-d::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,{GREEN},{CYAN}); }}
.kpi-e::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,{RED},{PINK}); }}
.kpi-icon  {{ font-size: 20px; margin-bottom: 10px; }}
.kpi-label {{ font-size: 10px; color: {TXT2}; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; }}
.kpi-value {{ font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: {TXT0}; margin: 5px 0 3px; }}
.kpi-sub   {{ font-size: 10px; color: #3a3a68; }}

.pg-hdr {{ display:flex; align-items:center; gap:16px; margin-bottom:1.4rem; padding-bottom:1rem; border-bottom:1px solid {BG3}; }}
.pg-icon {{ width:48px; height:48px; background:linear-gradient(135deg,{ACCENT},{CYAN}); border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:24px; flex-shrink:0; }}
.pg-title {{ font-size:22px; font-weight:700; color:{TXT0}; margin:0; line-height:1.2; }}
.pg-sub   {{ font-size:12px; color:{TXT2}; margin:3px 0 0; }}
.pg-badge {{ margin-left:auto; padding:5px 14px; background:{BG2}; border:1px solid {BG3}; border-radius:20px; font-size:10px; letter-spacing:0.06em; text-transform:uppercase; white-space:nowrap; font-weight:700; }}

.pred-box {{ background:linear-gradient(135deg,#0f0f22,#12122c); border:1px solid {CYAN}55; border-left:4px solid {CYAN}; border-radius:12px; padding:16px 20px; margin-top:14px; }}
.pred-lbl {{ font-size:10px; color:{CYAN}; text-transform:uppercase; letter-spacing:0.10em; margin-bottom:4px; font-weight:600; }}
.pred-val {{ font-size:32px; font-weight:700; font-family:'JetBrains Mono',monospace; color:{TXT0}; line-height:1.1; }}
.pred-note {{ font-size:11px; color:{TXT2}; margin-top:6px; line-height:1.7; }}

.sec-hdr {{ font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:{TXT2}; margin-bottom:1rem; padding-bottom:8px; border-bottom:1px solid {BG3}; }}
.chart-lbl {{ font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:{TXT2}; margin-bottom:6px; }}

[data-testid="stImage"] img {{ border-radius:12px; }}
.stAlert {{ border-radius:10px !important; font-size:12px !important; }}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# PLOTLY BASE LAYOUT
# ───────────────────────────────────────────────
PFONT = dict(family="Outfit, sans-serif", size=11, color=TXT1)

def plotly_base(height=280):
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG2,
        font=PFONT,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=BG3, zerolinecolor=BG3, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=BG3, zerolinecolor=BG3, tickfont=dict(size=10)),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=BG3, font_size=11, font_family="Outfit"),
    )

PLOTLY_CFG = {"displayModeBar": False, "responsive": True}

# ───────────────────────────────────────────────
# MATPLOTLIB THEME
# ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   BG1,
    "axes.facecolor":     BG2,
    "axes.edgecolor":     BG3,
    "axes.labelcolor":    TXT2,
    "axes.titlecolor":    TXT1,
    "xtick.color":        TXT2,
    "ytick.color":        TXT2,
    "text.color":         TXT0,
    "grid.color":         BG3,
    "grid.linewidth":     0.5,
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
})

def mpl_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

# ───────────────────────────────────────────────
# LOAD MODEL + DATA
# ───────────────────────────────────────────────
MODEL_PATH = "model.pkl"
EVAL_PATH  = "eval_results.pkl"
TRAIN_CSV  = "app_folder/train.csv.csv"
STORE_CSV  = "app_folder/store.csv"

for t in ["app_folder/train.csv.csv", "app_folder/train.csv", "train.csv.csv", "train.csv"]:
    if os.path.exists(t): TRAIN_CSV = t; break
for s in ["app_folder/store.csv", "store.csv"]:
    if os.path.exists(s): STORE_CSV = s; break

@st.cache_resource(show_spinner=False)
def load_model():
    bundle = results = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f: bundle = pickle.load(f)
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "rb") as f: results = pickle.load(f)
    return bundle, results

@st.cache_data(show_spinner=False)
def load_raw():
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(STORE_CSV)):
        return None
    train = pd.read_csv(TRAIN_CSV, dtype={"StateHoliday": str})
    store = pd.read_csv(STORE_CSV)
    df    = train.merge(store, on="Store", how="left")
    df["Open"] = df["Open"].fillna(1)
    df = df[df["Open"] == 1].copy()
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DOW"]   = df["Date"].dt.dayofweek
    return df

with st.spinner("Loading model & data …"):
    bundle, results = load_model()
    df = load_raw()

MODEL_READY = bundle is not None
HAS_DATA    = df is not None

# KPIs
if HAS_DATA:
    avg_sales  = int(df["Sales"].mean())
    n_stores   = df["Store"].nunique()
    promo_lift = round((df[df["Promo"]==1]["Sales"].mean() /
                        df[df["Promo"]==0]["Sales"].mean() - 1) * 100, 1)
else:
    avg_sales, n_stores, promo_lift = 5263, 1115, 18.0

ens_mae_v = f"₹{int(results['ens_mae']):,}" if results else "—"
ens_r2_v  = f"{results['ens_r2']:.4f}"       if results else "—"

# ───────────────────────────────────────────────
# SIDEBAR — PREDICTOR
# ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='margin-bottom:1rem'>
      <div style='font-size:18px;font-weight:700;color:{TXT0};'>📊 Sales Predictor</div>
      <div style='font-size:11px;color:{TXT2};margin-top:4px'>Daily sales forecast engine</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    store_id       = st.number_input("Store ID", min_value=1, max_value=1115, value=42)
    pred_date      = st.date_input("Forecast Date", datetime.date.today())
    promo          = st.selectbox("Promo Active?",   [1,0], format_func=lambda x:"✅ Yes" if x else "❌ No")
    state_holiday  = st.selectbox("State Holiday",   ["0","a","b","c"],
                                  format_func=lambda x:{"0":"None","a":"Public","b":"Easter","c":"Christmas"}[x])
    school_holiday = st.selectbox("School Holiday?", [0,1], format_func=lambda x:"✅ Yes" if x else "❌ No")

    st.divider()
    clicked = st.button("⚡  GENERATE FORECAST")

    if clicked:
        hmap = {"0":0,"a":1,"b":2,"c":3}
        yr, mo, dy = pred_date.year, pred_date.month, pred_date.day
        wk = pred_date.isocalendar()[1]

        if MODEL_READY:
            row = {"Store":store_id,"DayOfWeek":pred_date.weekday()+1,"Promo":promo,
                   "StateHoliday":hmap[state_holiday],"SchoolHoliday":school_holiday,
                   "Year":yr,"Month":mo,"Day":dy,"WeekOfYear":wk,
                   "IsWeekend":int(pred_date.weekday()>=5),"Quarter":(mo-1)//3+1,
                   "IsMonthStart":int(dy==1),
                   "IsMonthEnd":int(pd.Timestamp(pred_date).is_month_end),
                   "CompetitionDistance":500.0,"CompetitionMonthsOpen":24.0,
                   "PromoOpenWeeks":52.0,"StoreMeanSales":avg_sales,
                   "StoreMedianSales":avg_sales,"DayOfYear":pred_date.timetuple().tm_yday}
            fn = bundle["feature_names"]
            for f in fn:
                if f not in row: row[f] = avg_sales if "Sales" in f else 0
            inp  = pd.DataFrame([row])[fn]
            p1   = np.expm1(bundle["lgb"].predict(inp)[0])
            p2   = np.expm1(bundle["xgb"].predict(inp)[0])
            pred = int((p1 + p2) / 2)
        else:
            base = 4000 + (store_id % 5) * 400 + [0,200,100,300,500,900,-700][pred_date.weekday()]
            if promo:                       base *= 1.18
            if hmap[state_holiday] > 0:     base *= 0.72
            if school_holiday:              base *= 1.05
            pred = max(800, int(base + np.random.randint(-300, 300)))

        st.session_state["pred"]  = pred
        st.session_state["pmeta"] = dict(store=store_id,
                                          date=pred_date.strftime("%d %b %Y"),
                                          promo=promo, hol=state_holiday)

    if "pred" in st.session_state:
        m    = st.session_state["pmeta"]
        conf = "🟢 HIGH" if m["promo"] else ("🟡 MODERATE" if m["hol"] != "0" else "🔵 STANDARD")
        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-lbl">Predicted Daily Sales</div>
          <div class="pred-val">₹{st.session_state['pred']:,}</div>
          <div class="pred-note">Store {m['store']} &nbsp;·&nbsp; {m['date']}<br>
          Confidence &nbsp;<strong style="color:{CYAN}">{conf}</strong></div>
        </div>""", unsafe_allow_html=True)

    if not MODEL_READY:
        st.warning("⚠️ model.pkl not found!\nRun `python train.py` first.")

# ───────────────────────────────────────────────
# PAGE HEADER
# ───────────────────────────────────────────────
badge     = "✓ MODEL LOADED" if MODEL_READY else "⚠ RUN train.py FIRST"
badge_col = GREEN if MODEL_READY else RED
st.markdown(f"""
<div class="pg-hdr">
  <div class="pg-icon">📊</div>
  <div>
    <p class="pg-title">Chronic Retail Analytics</p>
    <p class="pg-sub">Sales Intelligence Platform &nbsp;·&nbsp; LightGBM + XGBoost Ensemble</p>
  </div>
  <div class="pg-badge" style="color:{badge_col};">{badge}</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# KPI STRIP
# ───────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-card kpi-a">
    <div class="kpi-icon">💰</div>
    <div class="kpi-label">Avg Daily Sales</div>
    <div class="kpi-value">₹{avg_sales:,}</div>
    <div class="kpi-sub">per store · open days</div>
  </div>
  <div class="kpi-card kpi-b">
    <div class="kpi-icon">🔖</div>
    <div class="kpi-label">Promo Lift</div>
    <div class="kpi-value">+{promo_lift}%</div>
    <div class="kpi-sub">revenue on promo days</div>
  </div>
  <div class="kpi-card kpi-c">
    <div class="kpi-icon">🏪</div>
    <div class="kpi-label">Stores Tracked</div>
    <div class="kpi-value">{n_stores:,}</div>
    <div class="kpi-sub">across all regions</div>
  </div>
  <div class="kpi-card kpi-d">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-label">Ensemble MAE</div>
    <div class="kpi-value">{ens_mae_v}</div>
    <div class="kpi-sub">mean absolute error</div>
  </div>
  <div class="kpi-card kpi-e">
    <div class="kpi-icon">📐</div>
    <div class="kpi-label">R² Score</div>
    <div class="kpi-value">{ens_r2_v}</div>
    <div class="kpi-sub">ensemble accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Trends", "🏪  Store Analysis", "🧠  Model Insights", "🔢  Data Explorer",
])


# ════════════════════════════════════════════════
# TAB 1 · TRENDS
# ════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-hdr">Sales Trends &amp; Time Patterns</div>', unsafe_allow_html=True)

    if HAS_DATA:
        monthly = (df.groupby(["Year","Month"])["Sales"].mean().reset_index()
                     .assign(Date=lambda d: pd.to_datetime(d.assign(Day=1)[["Year","Month","Day"]]))
                     .sort_values("Date"))
        promo_m = (df[df["Promo"]==1].groupby(["Year","Month"])["Sales"].mean().reset_index()
                     .assign(Date=lambda d: pd.to_datetime(d.assign(Day=1)[["Year","Month","Day"]]))
                     .sort_values("Date"))
        dow_avg = df.groupby("DOW")["Sales"].mean().reindex(range(7)).fillna(0)
    else:
        sales_d = [4820,5100,5380,5650,6100,5890,6250,6480,5920,5600,6100,7200]
        promo_d = [4600,4950,5200,5500,5900,5700,6050,6250,5700,5400,5900,6950]
        dow_avg = pd.Series([5100,5350,5200,5480,6100,6850,4200])

    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown('<div class="chart-lbl">Monthly Avg Sales Trend — Plotly Interactive</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if HAS_DATA:
            fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Sales"], name="All Sales",
                line=dict(color=ACCENT, width=2.5), fill="tozeroy",
                fillcolor="rgba(108,99,255,0.10)", mode="lines",
                hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=promo_m["Date"], y=promo_m["Sales"], name="Promo Days",
                line=dict(color=GREEN, width=1.8, dash="dash"), mode="lines",
                hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>"))
        else:
            x = list(range(12))
            fig.add_trace(go.Scatter(x=x, y=sales_d, name="Sales",
                line=dict(color=ACCENT, width=2.5), fill="tozeroy", fillcolor="rgba(108,99,255,0.10)"))
            fig.add_trace(go.Scatter(x=x, y=promo_d, name="Promo",
                line=dict(color=GREEN, width=1.8, dash="dash")))
            fig.update_xaxes(tickvals=x, ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"])
        fig.update_layout(**plotly_base(280),
            yaxis=dict(tickprefix="₹", tickformat=".2s", gridcolor=BG3),
            xaxis=dict(gridcolor=BG3))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    with c2:
        st.markdown('<div class="chart-lbl">Sales by Day of Week — Plotly Interactive</div>', unsafe_allow_html=True)
        vals   = dow_avg.values
        colors = [ACCENT if i < 5 else (GREEN if i == 5 else TXT2) for i in range(7)]
        fig2   = go.Figure(go.Bar(x=dow_labels, y=vals, marker_color=colors,
            marker_line_width=0, hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
        fig2.update_layout(**plotly_base(280),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickprefix="₹", tickformat=".2s", gridcolor=BG3), bargap=0.28)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CFG)

    # Heatmap — Matplotlib
    st.markdown('<div class="chart-lbl" style="margin-top:0.4rem">Sales Heatmap Weekday × Month — Matplotlib</div>', unsafe_allow_html=True)
    if HAS_DATA:
        heat = (df.assign(DOW=df["Date"].dt.dayofweek, Mon=df["Date"].dt.month)
                  .groupby(["DOW","Mon"])["Sales"].mean().unstack(fill_value=0).values)
    else:
        np.random.seed(42)
        heat = np.random.randint(4000,8000,size=(7,12)).astype(float)
        heat[5] *= 1.2; heat[6] *= 0.75
    fig_h, ax_h = plt.subplots(figsize=(14, 2.6))
    cmap_h = LinearSegmentedColormap.from_list("h", [BG3, ACCENT, CYAN])
    im = ax_h.imshow(heat, aspect="auto", cmap=cmap_h, interpolation="nearest")
    ax_h.set_xticks(range(heat.shape[1]))
    ax_h.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:heat.shape[1]], fontsize=9)
    ax_h.set_yticks(range(min(7, heat.shape[0])))
    ax_h.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:heat.shape[0]], fontsize=9)
    ax_h.grid(False)
    cb = fig_h.colorbar(im, ax=ax_h, fraction=0.012, pad=0.01)
    cb.ax.tick_params(colors=TXT2, labelsize=8); cb.outline.set_edgecolor(BG3)
    fig_h.tight_layout(pad=0.4)
    st.image(mpl_to_img(fig_h), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="chart-lbl">Promo vs Non-Promo — Plotly Interactive</div>', unsafe_allow_html=True)
        p_yes = df[df["Promo"]==1]["Sales"].mean() if HAS_DATA else 6200
        p_no  = df[df["Promo"]==0]["Sales"].mean() if HAS_DATA else 5100
        fig3  = go.Figure(go.Bar(x=["No Promo","Promo Active"], y=[p_no, p_yes],
            marker_color=[TXT2, GREEN], marker_line_width=0,
            text=[f"₹{p_no:,.0f}", f"₹{p_yes:,.0f}"], textposition="outside",
            textfont=dict(size=11, color=TXT1),
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
        fig3.update_layout(**plotly_base(240),
            yaxis=dict(tickprefix="₹", tickformat=".2s", gridcolor=BG3),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"), bargap=0.5)
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CFG)

    with c4:
        st.markdown('<div class="chart-lbl">Year-over-Year — Plotly Interactive</div>', unsafe_allow_html=True)
        yearly = df.groupby("Year")["Sales"].mean().reset_index() if HAS_DATA else pd.DataFrame({"Year":[2013,2014,2015],"Sales":[5100,5800,6300]})
        fig4   = go.Figure(go.Bar(x=yearly["Year"].astype(str), y=yearly["Sales"],
            marker_color=PAL[:len(yearly)], marker_line_width=0,
            text=[f"₹{v:,.0f}" for v in yearly["Sales"]], textposition="outside",
            textfont=dict(size=11, color=TXT1),
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
        fig4.update_layout(**plotly_base(240),
            yaxis=dict(tickprefix="₹", tickformat=".2s", gridcolor=BG3),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"), bargap=0.4)
        st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CFG)


# ════════════════════════════════════════════════
# TAB 2 · STORE ANALYSIS
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-hdr">Store Performance &amp; Segmentation</div>', unsafe_allow_html=True)

    if HAS_DATA and "StoreType" in df.columns:
        st_data  = df.groupby("StoreType")["Sales"].mean().reset_index()
        ass_data = (df.groupby("Assortment")
                      .agg(stores=("Store","nunique"), sales=("Sales","mean")).reset_index())
        ass_map  = {"a":"Basic","b":"Extra","c":"Extended"}
        ass_data["Assortment"] = ass_data["Assortment"].map(ass_map).fillna(ass_data["Assortment"])
        cdf = (df.groupby("Store").agg(AvgSales=("Sales","mean"),
                CompDist=("CompetitionDistance","first")).dropna()
                .sample(min(500, 1000), random_state=7))
    else:
        st_data  = pd.DataFrame({"StoreType":["a","b","c","d"],"Sales":[6100,4700,7200,5300]})
        ass_data = pd.DataFrame({"Assortment":["Basic","Extra","Extended"],"stores":[53,9,38],"sales":[5100,4800,6200]})
        np.random.seed(7)
        cd  = np.random.exponential(2000, 300)
        cs  = np.clip(7000 - cd*0.4 + np.random.normal(0,800,300), 1500, 9000)
        cdf = pd.DataFrame({"CompDist":cd, "AvgSales":cs})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-lbl">Avg Sales by Store Type — Plotly Interactive</div>', unsafe_allow_html=True)
        fig5 = go.Figure(go.Bar(
            x=[f"Type {t.upper()}" for t in st_data["StoreType"]], y=st_data["Sales"],
            marker_color=PAL[:len(st_data)], marker_line_width=0,
            text=[f"₹{v:,.0f}" for v in st_data["Sales"]], textposition="outside",
            textfont=dict(size=11, color=TXT1),
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
        fig5.update_layout(**plotly_base(270),
            yaxis=dict(tickprefix="₹", tickformat=".2s", gridcolor=BG3),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"), bargap=0.35)
        st.plotly_chart(fig5, use_container_width=True, config=PLOTLY_CFG)

    with c2:
        st.markdown('<div class="chart-lbl">Assortment Distribution — Plotly Interactive</div>', unsafe_allow_html=True)
        fig6 = go.Figure(go.Pie(labels=ass_data["Assortment"], values=ass_data["stores"],
            hole=0.55, marker=dict(colors=PAL[:len(ass_data)], line=dict(color=BG1, width=3)),
            textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="%{label}: %{value} stores (%{percent})<extra></extra>"))
        fig6.update_layout(height=270, paper_bgcolor="rgba(0,0,0,0)",
            font=PFONT, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
        st.plotly_chart(fig6, use_container_width=True, config=PLOTLY_CFG)

    # Competition scatter — Matplotlib
    st.markdown('<div class="chart-lbl" style="margin-top:0.4rem">Competition Distance vs Avg Sales — Matplotlib</div>', unsafe_allow_html=True)
    fig_c, ax_c = plt.subplots(figsize=(14, 3.5))
    sc = ax_c.scatter(cdf["CompDist"], cdf["AvgSales"],
        c=cdf["AvgSales"], cmap=LinearSegmentedColormap.from_list("g",[ACCENT,CYAN,GREEN]),
        s=20, alpha=0.55, edgecolors="none", zorder=3)
    z  = np.polyfit(cdf["CompDist"], cdf["AvgSales"], 1)
    xl = np.linspace(cdf["CompDist"].min(), cdf["CompDist"].max(), 200)
    ax_c.plot(xl, np.poly1d(z)(xl), color=AMBER, lw=2, ls="--", alpha=0.8, label="Trend")
    ax_c.set_xlabel("Competition Distance (m)", fontsize=9)
    ax_c.set_ylabel("Avg Daily Sales (₹)", fontsize=9)
    ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}k"))
    ax_c.legend(fontsize=9, labelcolor=TXT1, facecolor=BG2, edgecolor=BG3)
    cb_c = fig_c.colorbar(sc, ax=ax_c, fraction=0.012, pad=0.01)
    cb_c.set_label("Avg Sales (₹)", color=TXT2, fontsize=8)
    cb_c.ax.tick_params(colors=TXT2, labelsize=8); cb_c.outline.set_edgecolor(BG3)
    fig_c.tight_layout(pad=0.4)
    st.image(mpl_to_img(fig_c), use_container_width=True)

    if HAS_DATA:
        st.divider()
        st.markdown('<div class="sec-hdr">Top &amp; Bottom Stores by Avg Sales</div>', unsafe_allow_html=True)
        sp = df.groupby("Store")["Sales"].mean().sort_values(ascending=False)
        ca, cb2 = st.columns(2)
        with ca:
            st.caption("🏆 Top 10 Stores")
            t10 = sp.head(10).reset_index(); t10.columns = ["Store ID","Avg Sales"]
            t10["Avg Sales"] = t10["Avg Sales"].apply(lambda v: f"₹{v:,.0f}")
            st.dataframe(t10, use_container_width=True, hide_index=True)
        with cb2:
            st.caption("📉 Bottom 10 Stores")
            b10 = sp.tail(10).reset_index(); b10.columns = ["Store ID","Avg Sales"]
            b10["Avg Sales"] = b10["Avg Sales"].apply(lambda v: f"₹{v:,.0f}")
            st.dataframe(b10, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# TAB 3 · MODEL INSIGHTS
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-hdr">Model Performance &amp; Explainability</div>', unsafe_allow_html=True)

    if results:
        # Model comparison — Plotly
        st.markdown('<div class="chart-lbl">Model Comparison — Plotly Interactive</div>', unsafe_allow_html=True)
        fig_mc = make_subplots(rows=1, cols=2,
            subplot_titles=["Mean Absolute Error (lower = better)", "R² Score (higher = better)"])
        models = ["LightGBM","XGBoost","Ensemble"]
        mae_v  = [results["lgb_mae"], results["xgb_mae"], results["ens_mae"]]
        r2_v   = [results["lgb_r2"],  results["xgb_r2"],  results["ens_r2"]]
        fig_mc.add_trace(go.Bar(x=models, y=mae_v, marker_color=[ACCENT,CYAN,GREEN],
            marker_line_width=0, text=[f"₹{int(v):,}" for v in mae_v],
            textposition="outside", textfont=dict(size=11, color=TXT1),
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>", showlegend=False), row=1, col=1)
        fig_mc.add_trace(go.Bar(x=models, y=r2_v, marker_color=[ACCENT,CYAN,GREEN],
            marker_line_width=0, text=[f"{v:.4f}" for v in r2_v],
            textposition="outside", textfont=dict(size=11, color=TXT1),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>", showlegend=False), row=1, col=2)
        fig_mc.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG2,
            font=PFONT, margin=dict(l=10,r=10,t=40,b=10), bargap=0.4)
        for ann in fig_mc.layout.annotations:
            ann.font.color = TXT2; ann.font.size = 10
        fig_mc.update_xaxes(gridcolor="rgba(0,0,0,0)")
        fig_mc.update_yaxes(gridcolor=BG3)
        fig_mc.update_yaxes(tickprefix="₹", tickformat=".2s", row=1, col=1)
        st.plotly_chart(fig_mc, use_container_width=True, config=PLOTLY_CFG)

        c1, c2 = st.columns([1, 1.2])
        with c1:
            # Feature importance — Plotly
            st.markdown('<div class="chart-lbl">Feature Importances — Plotly Interactive</div>', unsafe_allow_html=True)
            imp  = results["importance"].head(15)
            norm = (imp / imp.max() * 100).round(1)
            fig_fi = go.Figure(go.Bar(x=norm.values, y=norm.index, orientation="h",
                marker=dict(color=norm.values, colorscale=[[0,BG4],[0.5,ACCENT],[1,CYAN]], line_width=0),
                text=[f"{v:.0f}%" for v in norm.values], textposition="outside",
                textfont=dict(size=10, color=TXT1),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            fig_fi.update_layout(height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG2,
                font=PFONT, margin=dict(l=10,r=50,t=10,b=10),
                xaxis=dict(range=[0,130], gridcolor=BG3, ticksuffix="%"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CFG)

        with c2:
            # Actual vs Predicted — Matplotlib
            st.markdown('<div class="chart-lbl">Actual vs Predicted — Matplotlib (error colour)</div>', unsafe_allow_html=True)
            actual_v = results["actual"]; pred_v = results["ens_pred"]
            fig_ap, ax_ap = plt.subplots(figsize=(5.5, 5))
            sc_ap = ax_ap.scatter(actual_v, pred_v,
                c=np.abs(pred_v - actual_v),
                cmap=LinearSegmentedColormap.from_list("e",[GREEN,AMBER,RED]),
                s=18, alpha=0.6, edgecolors="none", zorder=3)
            dm = max(actual_v.max(), pred_v.max())
            ax_ap.plot([0,dm],[0,dm], color=ACCENT, lw=1.5, ls="--", alpha=0.8, label="Perfect Fit")
            ax_ap.set_xlabel("Actual Sales (₹)", fontsize=9)
            ax_ap.set_ylabel("Predicted Sales (₹)", fontsize=9)
            ax_ap.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}k"))
            ax_ap.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}k"))
            ax_ap.legend(fontsize=9, labelcolor=TXT1, facecolor=BG2, edgecolor=BG3)
            cb_ap = fig_ap.colorbar(sc_ap, ax=ax_ap, fraction=0.04, pad=0.01)
            cb_ap.set_label("|Error| ₹", color=TXT2, fontsize=8)
            cb_ap.ax.tick_params(colors=TXT2, labelsize=8); cb_ap.outline.set_edgecolor(BG3)
            fig_ap.tight_layout(pad=0.5)
            st.image(mpl_to_img(fig_ap), use_container_width=True)

        # Residuals — Plotly
        st.markdown('<div class="chart-lbl" style="margin-top:0.4rem">Residuals Distribution — Plotly Interactive</div>', unsafe_allow_html=True)
        residuals = pred_v - actual_v
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(x=residuals, nbinsx=60, marker_color=ACCENT,
            marker_line_width=0, opacity=0.8,
            hovertemplate="Error ₹%{x:,.0f}: %{y} samples<extra></extra>", name="Residuals"))
        fig_res.add_vline(x=0, line_dash="dash", line_color=GREEN, line_width=1.5)
        fig_res.add_annotation(x=0, y=0.95, yref="paper", text="Zero Error",
            showarrow=False, font=dict(color=GREEN, size=10), xshift=30)
        fig_res.update_layout(**plotly_base(230),
            xaxis=dict(title="Prediction Error (₹)", tickprefix="₹", gridcolor=BG3),
            yaxis=dict(title="Count", gridcolor=BG3))
        st.plotly_chart(fig_res, use_container_width=True, config=PLOTLY_CFG)

        st.divider()
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("LightGBM MAE", f"₹{results['lgb_mae']:,.0f}")
        m2.metric("LightGBM R²",  f"{results['lgb_r2']:.4f}")
        m3.metric("XGBoost MAE",  f"₹{results['xgb_mae']:,.0f}")
        m4.metric("XGBoost R²",   f"{results['xgb_r2']:.4f}")
        m5.metric("Ensemble MAE", f"₹{results['ens_mae']:,.0f}",
                  delta=f"↓{results['lgb_mae']-results['ens_mae']:,.0f} vs LGB")
        m6.metric("Ensemble R²",  f"{results['ens_r2']:.4f}",
                  delta=f"+{results['ens_r2']-results['lgb_r2']:.4f} vs LGB")
    else:
        st.error("⚠️ No model results found. Please run `python train.py` first!")


# ════════════════════════════════════════════════
# TAB 4 · DATA EXPLORER
# ════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-hdr">Raw Data Explorer</div>', unsafe_allow_html=True)

    if HAS_DATA:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total Records",  f"{len(df):,}")
        m2.metric("Unique Stores",  f"{df['Store'].nunique():,}")
        m3.metric("Date Range",     f"{df['Date'].dt.year.min()}–{df['Date'].dt.year.max()}")
        m4.metric("Avg Customers",  f"{int(df['Customers'].mean()):,}" if "Customers" in df.columns else "—")

        st.divider()

        if "StoreType" in df.columns:
            st.markdown('<div class="chart-lbl">Sales Distribution by Store Type — Plotly Box Plot</div>', unsafe_allow_html=True)
            fig_box = go.Figure()
            for i, st_type in enumerate(sorted(df["StoreType"].unique())):
                sub = df[df["StoreType"]==st_type]["Sales"].sample(min(5000,len(df[df["StoreType"]==st_type])), random_state=1)
                r,g,b = int(PAL[i][1:3],16), int(PAL[i][3:5],16), int(PAL[i][5:7],16)
                fig_box.add_trace(go.Box(y=sub, name=f"Type {st_type.upper()}",
                    marker_color=PAL[i], line_color=PAL[i],
                    fillcolor=f"rgba({r},{g},{b},0.15)",
                    hovertemplate=f"Type {st_type.upper()}<br>₹%{{y:,.0f}}<extra></extra>"))
            fig_box.update_layout(**plotly_base(300),
                yaxis=dict(tickprefix="₹", tickformat=".2s", gridcolor=BG3, title="Sales"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CFG)

        st.divider()
        show_cols = [c for c in ["Store","DayOfWeek","Sales","Customers","Promo",
                                  "StateHoliday","SchoolHoliday","Year","Month","Day"]
                     if c in df.columns]
        st.caption("📋 Data Preview — first 500 rows")
        st.dataframe(df[show_cols].head(500), use_container_width=True, hide_index=True)

        st.divider()
        st.caption("📊 Descriptive Statistics")
        st.dataframe(df[show_cols].describe().T.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("📂 Place train.csv & store.csv in app_folder/ to explore data.")
