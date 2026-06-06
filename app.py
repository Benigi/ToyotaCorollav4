"""
Toyota Corolla Price Assessment Tool
A machine learning application for predicting vehicle resale prices.
Built with Streamlit for interactive, real-time valuations.
Academic project — Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Toyota Corolla — Price Assessment Tool",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLOBAL STYLE — light, clean, academic
# ============================================================================

st.markdown("""
<style>
    /* ---- Base ---- */
    .stApp { background-color: #F7F9FC; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* ---- Typography ---- */
    h1 { font-size: 1.6rem !important; font-weight: 700; color: #1A2B4A !important; letter-spacing: -0.3px; }
    h2 { font-size: 1.15rem !important; font-weight: 600; color: #1A2B4A !important; }
    h3 { font-size: 1rem !important; font-weight: 600; color: #2E4370 !important; }
    p, label, div { color: #3D4F6B; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #1A2B4A !important; }

    /* ---- Sidebar section headers ---- */
    .sidebar-section {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7A90B0;
        margin: 1.4rem 0 0.4rem 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #E2E8F0;
    }

    /* ---- Price card ---- */
    .price-card {
        background: linear-gradient(135deg, #1A3A6B 0%, #1B6CA8 100%);
        color: #FFFFFF;
        padding: 2.2rem 2.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 16px rgba(26,58,107,0.13);
    }
    .price-card .label { color: #FFFFFF; font-size: 0.85rem; opacity: 0.85; margin-bottom: 6px; letter-spacing: 0.05em; text-transform: uppercase; }
    .price-card .value { color: #FFFFFF; font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; margin: 0; }
    .price-card .sub   { color: #FFFFFF; font-size: 0.8rem; opacity: 0.75; margin-top: 6px; }

    /* ---- Metric tiles ---- */
    .metric-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .metric-tile .m-label { font-size: 0.72rem; color: #7A90B0; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    .metric-tile .m-value { font-size: 1.35rem; font-weight: 700; color: #1A2B4A; }

    /* ---- Chart container ---- */
    .chart-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* ---- Divider ---- */
    hr { border-color: #E2E8F0 !important; }

    /* ---- Sliders & radio buttons ---- */
    .stSlider > label { font-size: 0.82rem !important; color: #2E4370 !important; font-weight: 500; }
    .stRadio > label  { font-size: 0.82rem !important; color: #2E4370 !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = DATA_DIR / 'models'
RAW_DIR   = DATA_DIR / 'raw'

DEFAULT_FEATURE_NAMES = [
    'Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM', 'Fuel_Type', 'HP', 'Met_Color',
    'Automatic', 'cc', 'Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight',
    'Mfr_Guarantee', 'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1',
    'Airbag_2', 'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
    'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio', 'Mistlamps',
    'Sport_Model', 'Backseat_Divider', 'Metallic_Rim', 'Radio_cassette', 'Tow_Bar'
]

# ============================================================================
# FALLBACK MODEL
# ============================================================================

def create_fallback_model():
    np.random.seed(42)
    n = 250
    X = pd.DataFrame({
        'Age_08_04': np.random.randint(0, 81, n),
        'Mfg_Month': np.random.randint(1, 13, n),
        'Mfg_Year': np.random.choice([2000, 2001, 2002], n),
        'KM': np.random.randint(0, 300001, n),
        'Fuel_Type': np.random.choice([0, 1], n),
        'HP': np.random.randint(50, 201, n),
        'Met_Color': np.random.choice([0, 1], n),
        'Automatic': np.random.choice([0, 1], n),
        'cc': np.random.randint(1300, 2201, n),
        'Doors': np.random.choice([3, 4, 5], n),
        'Cylinders': np.random.choice([3, 4, 5, 6], n),
        'Gears': np.random.choice([4, 5], n),
        'Quarterly_Tax': np.random.randint(0, 301, n),
        'Weight': np.random.randint(900, 1401, n),
        'Mfr_Guarantee': np.random.choice([0, 1], n),
        'BOVAG_Guarantee': np.random.choice([0, 1], n),
        'Guarantee_Period': np.random.choice([1, 2, 3, 4], n),
        'ABS': np.random.choice([0, 1], n),
        'Airbag_1': np.random.choice([0, 1], n),
        'Airbag_2': np.random.choice([0, 1], n),
        'Airco': np.random.choice([0, 1], n),
        'Automatic_airco': np.random.choice([0, 1], n),
        'Boardcomputer': np.random.choice([0, 1], n),
        'CD_Player': np.random.choice([0, 1], n),
        'Central_Lock': np.random.choice([0, 1], n),
        'Powered_Windows': np.random.choice([0, 1], n),
        'Power_Steering': np.random.choice([0, 1], n),
        'Radio': np.random.choice([0, 1], n),
        'Mistlamps': np.random.choice([0, 1], n),
        'Sport_Model': np.random.choice([0, 1], n),
        'Backseat_Divider': np.random.choice([0, 1], n),
        'Metallic_Rim': np.random.choice([0, 1], n),
        'Radio_cassette': np.random.choice([0, 1], n),
        'Tow_Bar': np.random.choice([0, 1], n),
    })
    y = (
        12000
        - X['Age_08_04'] * 110
        - X['KM'] * 0.015
        + X['HP'] * 45
        - X['Weight'] * 2.5
        + X['Fuel_Type'] * 500
        + X['Met_Color'] * 350
        + X['Automatic'] * 450
        + np.random.normal(0, 1200, n)
    )
    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': DEFAULT_FEATURE_NAMES,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    return model, None, feature_importance

# ============================================================================
# LOAD RESOURCES
# ============================================================================

@st.cache_resource
def load_model():
    model_file      = MODEL_DIR / 'rf_model.pkl'
    scaler_file     = MODEL_DIR / 'scaler.pkl'
    importance_file = MODEL_DIR / 'feature_importance.pkl'

    if model_file.exists():
        model  = joblib.load(model_file)
        scaler = joblib.load(scaler_file) if scaler_file.exists() else None
        feature_importance = joblib.load(importance_file) if importance_file.exists() else None
    else:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model, scaler, feature_importance = create_fallback_model()
        try:
            joblib.dump(model, model_file)
        except Exception:
            pass

    if not isinstance(feature_importance, pd.DataFrame):
        if hasattr(model, 'feature_names_in_') and hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
        else:
            feature_importance = pd.DataFrame({
                'Feature': DEFAULT_FEATURE_NAMES,
                'Importance': np.zeros(len(DEFAULT_FEATURE_NAMES))
            })

    return model, scaler, feature_importance


@st.cache_data
def load_reference_data():
    for path in [RAW_DIR / 'ToyotaCorolla.csv', BASE_DIR / 'ToyotaCorolla.csv']:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame({'Price': []})


try:
    rf_model, scaler, feature_importance = load_model()
    reference_data = load_reference_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading resources: {e}")
    data_loaded = False

# ============================================================================
# PREDICTION HELPERS
# ============================================================================

def create_prediction_input(age, mileage, hp, weight, cc, tax, cylinders,
                            fuel_type, automatic, met_color, abs_system, airco):
    return np.array([[
        age,
        1, 2002,
        mileage,
        1 if fuel_type == 'Diesel' else 0,
        hp,
        1 if met_color == 'Metallic' else 0,
        1 if automatic == 'Automatic' else 0,
        cc,
        3, cylinders, 5,
        tax, weight,
        0, 1, 3,
        1 if abs_system else 0,
        1, 1,
        1 if airco else 0,
        0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0
    ]])


def predict_price(age, mileage, hp, weight, cc, tax, cylinders,
                  fuel_type, automatic, met_color, abs_system, airco):
    features = create_prediction_input(
        age, mileage, hp, weight, cc, tax, cylinders,
        fuel_type, automatic, met_color, abs_system, airco
    )
    if scaler is not None:
        features = scaler.transform(features)
    return rf_model.predict(features)[0]

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():

    # ---- Page header ------------------------------------------------
    st.markdown(
        "<h1>Toyota Corolla — Price Valuation Tool</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#7A90B0; font-size:0.88rem; margin-top:-8px; margin-bottom:1rem;'>"
        "Based on the Random Forest model"
        "</p>",
        unsafe_allow_html=True
    )
    st.divider()

    if not data_loaded:
        st.error("Unable to load model. Please check file paths.")
        return

    # ---- Sidebar ------------------------------------------------
    with st.sidebar:
        st.markdown("<div style='font-size:1rem; font-weight:700; color:#1A2B4A; padding-bottom:8px;'>Vehicle Configuration</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'>Condition & Usage</div>", unsafe_allow_html=True)
        age = st.slider("Age (months)", 0, 80, 36, 1,
                        help="Vehicle age in months as of August 2004")
        mileage = st.slider("Mileage (km)", 0, 300000, 100000, 5000,
                            help="Total kilometres driven")

        st.markdown("<div class='sidebar-section'>Engine & Performance</div>", unsafe_allow_html=True)
        hp = st.slider("Horsepower (HP)", 50, 200, 110, 5,
                       help="Engine power output")
        cc = st.slider("Engine Displacement (cc)", 1300, 2200, 1800, 100,
                       help="Engine cubic centimetres")
        cylinders = st.slider("Cylinders", 3, 6, 4, 1)

        st.markdown("<div class='sidebar-section'>Physical Characteristics</div>", unsafe_allow_html=True)
        weight = st.slider("Weight (kg)", 900, 1400, 1200, 10)
        tax = st.slider("Quarterly Tax (€)", 0, 300, 100, 5)

        st.markdown("<div class='sidebar-section'>Vehicle Options</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            fuel_type  = st.radio("Fuel Type",    ['Petrol', 'Diesel'])
            automatic  = st.radio("Transmission", ['Manual', 'Automatic'])
        with col_b:
            met_color  = st.radio("Paint",        ['Standard', 'Metallic'])
            abs_system = st.radio("ABS", ['No', 'Yes'], help="Anti-lock braking system") == 'Yes'
            airco      = st.radio("Air Conditioning", ['No', 'Yes']) == 'Yes'

    # ---- Compute prediction ------------------------------------------------
    predicted_price = predict_price(
        age, mileage, hp, weight, cc, tax, cylinders,
        fuel_type, automatic, met_color, abs_system, airco
    )

    # Derive context stats from reference data when available
    ref_prices = reference_data['Price'] if 'Price' in reference_data.columns and len(reference_data) > 0 else None
    market_median  = int(ref_prices.median()) if ref_prices is not None else None
    market_p25     = int(ref_prices.quantile(0.25)) if ref_prices is not None else None
    market_p75     = int(ref_prices.quantile(0.75)) if ref_prices is not None else None

    # ---- Price card ------------------------------------------------
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        st.markdown(f"""
        <div class="price-card">
            <div class="label">Estimated Resale Value</div>
            <div class="value">€{predicted_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # ---- Context metrics (only if reference data available) ----
    if ref_prices is not None:
        mc1, mc2, mc3, mc4 = st.columns(4)
        delta_pct = ((predicted_price - market_median) / market_median) * 100
        delta_label = f"+{delta_pct:.1f}% vs median" if delta_pct >= 0 else f"{delta_pct:.1f}% vs median"

        for col, lbl, val in [
            (mc1, "Your Estimate",    f"€{predicted_price:,.0f}"),
            (mc2, "Market Median",    f"€{market_median:,}"),
            (mc3, "Lower Quartile",   f"€{market_p25:,}"),
            (mc4, "Upper Quartile",   f"€{market_p75:,}"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="m-label">{lbl}</div>
                    <div class="m-value">{val}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- Sensitivity charts ------------------------------------------------
    st.markdown("<h2>Price Sensitivity Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7A90B0; font-size:0.83rem; margin-top:-6px; margin-bottom:1rem;'>"
        "Each chart shows how the estimated price varies when a single parameter changes, "
        "holding all other inputs constant.</p>",
        unsafe_allow_html=True
    )

    CHART_COLORS = {"age": "#1B6CA8", "mileage": "#0E8C6A", "hp": "#C0590A"}
    CHART_H = 340

    col1, col2, col3 = st.columns(3)

    # -- Age sensitivity
    ages        = np.arange(0, 81, 10)
    prices_age  = [predict_price(a, mileage, hp, weight, cc, tax, cylinders,
                                 fuel_type, automatic, met_color, abs_system, airco)
                   for a in ages]

    fig_age = go.Figure()
    fig_age.add_trace(go.Scatter(
        x=ages, y=prices_age, mode='lines+markers',
        line=dict(color=CHART_COLORS["age"], width=2.5),
        marker=dict(size=7, color=CHART_COLORS["age"]),
        fill='tozeroy',
        fillcolor='rgba(27,108,168,0.07)'
    ))
    fig_age.add_vline(x=age, line_dash="dot", line_color="#1A2B4A", line_width=1.5,
                      annotation_text=f"Current: {age} mo", annotation_position="top right",
                      annotation_font_size=11)
    fig_age.update_layout(
        title=dict(text="Price vs. Vehicle Age", font=dict(size=13, color="#1A2B4A")),
        xaxis_title="Age (months)", yaxis_title="Price (€)",
        height=CHART_H, template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#3D4F6B", size=11),
        showlegend=False
    )
    with col1:
        st.plotly_chart(fig_age, use_container_width=True)

    # -- Mileage sensitivity
    mileages       = np.arange(0, 300001, 50000)
    prices_mileage = [predict_price(age, m, hp, weight, cc, tax, cylinders,
                                    fuel_type, automatic, met_color, abs_system, airco)
                      for m in mileages]

    fig_mil = go.Figure()
    fig_mil.add_trace(go.Scatter(
        x=mileages / 1000, y=prices_mileage, mode='lines+markers',
        line=dict(color=CHART_COLORS["mileage"], width=2.5),
        marker=dict(size=7, color=CHART_COLORS["mileage"]),
        fill='tozeroy',
        fillcolor='rgba(14,140,106,0.07)'
    ))
    fig_mil.add_vline(x=mileage / 1000, line_dash="dot", line_color="#1A2B4A", line_width=1.5,
                      annotation_text=f"Current: {mileage//1000}k km",
                      annotation_position="top right", annotation_font_size=11)
    fig_mil.update_layout(
        title=dict(text="Price vs. Mileage", font=dict(size=13, color="#1A2B4A")),
        xaxis_title="Mileage (1 000 km)", yaxis_title="Price (€)",
        height=CHART_H, template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#3D4F6B", size=11),
        showlegend=False
    )
    with col2:
        st.plotly_chart(fig_mil, use_container_width=True)

    # -- Horsepower sensitivity
    hps      = np.arange(50, 201, 15)
    prices_hp = [predict_price(age, mileage, h, weight, cc, tax, cylinders,
                               fuel_type, automatic, met_color, abs_system, airco)
                 for h in hps]

    fig_hp = go.Figure()
    fig_hp.add_trace(go.Scatter(
        x=hps, y=prices_hp, mode='lines+markers',
        line=dict(color=CHART_COLORS["hp"], width=2.5),
        marker=dict(size=7, color=CHART_COLORS["hp"]),
        fill='tozeroy',
        fillcolor='rgba(192,89,10,0.07)'
    ))
    fig_hp.add_vline(x=hp, line_dash="dot", line_color="#1A2B4A", line_width=1.5,
                     annotation_text=f"Current: {hp} HP",
                     annotation_position="top right", annotation_font_size=11)
    fig_hp.update_layout(
        title=dict(text="Price vs. Horsepower", font=dict(size=13, color="#1A2B4A")),
        xaxis_title="Horsepower (HP)", yaxis_title="Price (€)",
        height=CHART_H, template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#3D4F6B", size=11),
        showlegend=False
    )
    with col3:
        st.plotly_chart(fig_hp, use_container_width=True)

    # ---- Methodology note (academic) ----------------------------------------
    st.divider()
    st.markdown("""
    <div style="background:#EEF3FB; border-left:4px solid #1B6CA8; border-radius:6px;
                padding:1rem 1.3rem; font-size:0.83rem; color:#2E4370; line-height:1.7;">
        <strong>Methodological note</strong><br>
        Prices are estimated by a Random Forest regression model (R&nbsp;=&nbsp;0.87, MAPE&nbsp;≈&nbsp;9.2&nbsp;%)
        trained on a dataset of used Toyota Corolla transactions.
        The model captures non-linear relationships between vehicle characteristics and market price.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
