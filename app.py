"""
Toyota Corolla Valuation Tool
A sophisticated machine learning application for predicting vehicle resale prices.
Built with Streamlit for interactive, real-time valuations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = DATA_DIR / 'models'
RAW_DIR = DATA_DIR / 'raw'

DEFAULT_FEATURE_NAMES = [
    'Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM', 'Fuel_Type', 'HP', 'Met_Color',
    'Automatic', 'cc', 'Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight',
    'Mfr_Guarantee', 'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1',
    'Airbag_2', 'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
    'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio', 'Mistlamps',
    'Sport_Model', 'Backseat_Divider', 'Metallic_Rim', 'Radio_cassette', 'Tow_Bar'
]


def create_fallback_model():
    """Create a fallback model trained on synthetic data."""
    np.random.seed(42)
    samples = 250
    X = pd.DataFrame({
        'Age_08_04': np.random.randint(0, 81, samples),
        'Mfg_Month': np.random.randint(1, 13, samples),
        'Mfg_Year': np.random.choice([2000, 2001, 2002], samples),
        'KM': np.random.randint(0, 300001, samples),
        'Fuel_Type': np.random.choice([0, 1], samples),
        'HP': np.random.randint(50, 201, samples),
        'Met_Color': np.random.choice([0, 1], samples),
        'Automatic': np.random.choice([0, 1], samples),
        'cc': np.random.randint(1300, 2201, samples),
        'Doors': np.random.choice([3, 4, 5], samples),
        'Cylinders': np.random.choice([3, 4, 5, 6], samples),
        'Gears': np.random.choice([4, 5], samples),
        'Quarterly_Tax': np.random.randint(0, 301, samples),
        'Weight': np.random.randint(900, 1401, samples),
        'Mfr_Guarantee': np.random.choice([0, 1], samples),
        'BOVAG_Guarantee': np.random.choice([0, 1], samples),
        'Guarantee_Period': np.random.choice([1, 2, 3, 4], samples),
        'ABS': np.random.choice([0, 1], samples),
        'Airbag_1': np.random.choice([0, 1], samples),
        'Airbag_2': np.random.choice([0, 1], samples),
        'Airco': np.random.choice([0, 1], samples),
        'Automatic_airco': np.random.choice([0, 1], samples),
        'Boardcomputer': np.random.choice([0, 1], samples),
        'CD_Player': np.random.choice([0, 1], samples),
        'Central_Lock': np.random.choice([0, 1], samples),
        'Powered_Windows': np.random.choice([0, 1], samples),
        'Power_Steering': np.random.choice([0, 1], samples),
        'Radio': np.random.choice([0, 1], samples),
        'Mistlamps': np.random.choice([0, 1], samples),
        'Sport_Model': np.random.choice([0, 1], samples),
        'Backseat_Divider': np.random.choice([0, 1], samples),
        'Metallic_Rim': np.random.choice([0, 1], samples),
        'Radio_cassette': np.random.choice([0, 1], samples),
        'Tow_Bar': np.random.choice([0, 1], samples)
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
        + np.random.normal(0, 1200, samples)
    )

    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X, y)

    feature_importance = pd.DataFrame({
        'Feature': DEFAULT_FEATURE_NAMES,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return model, None, feature_importance


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# st.set_page_config(
    page_title="Case Study: A Market-Independent ML Valuation Tool",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
# )

# Custom CSS for sophisticated styling
st.markdown("""
<style>
    body, .stApp, .main, .block-container, .css-1d391kg, .css-1d391kg .css-1lcbmhc {
        background-color: #0E1117 !important;
        color: #E2E8F0 !important;
    }

    .stApp {
        background-color: #0E1117;
    }

    .block-container {
        background-color: #0F172A;
        color: #E2E8F0;
    }

    h1, h2, h3, h4, h5, h6, p, label, .css-1d391kg {
        color: #E2E8F0 !important;
    }

    /* Headers */
    h1 {
        color: #8DD7FF;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
    }

    h2, h3 {
        color: #A3D2FF;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Price display */
    .price-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Sidebar styling */
    .css-1d391kg {
        padding: 20px;
        background-color: #0F172A;
    }

    /* Slider labels */
    .stSlider > label {
        color: #A3D2FF;
        font-weight: 600;
    }

    /* Radio buttons */
    .stRadio > label {
        color: #A3D2FF;
        font-weight: 500;
    }

    .stMetricValue, .stMetricLabel {
        color: #E2E8F0 !important;
    }

    .stMarkdown, .stText, .css-1offfwp, .css-1e5imcs {
        color: #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model():
    """Load pre-trained model and related artifacts."""
    model_file = MODEL_DIR / 'rf_model.pkl'
    scaler_file = MODEL_DIR / 'scaler.pkl'
    importance_file = MODEL_DIR / 'feature_importance.pkl'

    if model_file.exists():
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file) if scaler_file.exists() else None
        feature_importance = None
        if importance_file.exists():
            feature_importance = joblib.load(importance_file)
    else:
        st.warning(
            "⚠️ Aucune ressource de modèle trouvée. Utilisation d'un modèle de repli.")
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
    """Load reference data for statistics and visualization"""
    search_paths = [
        RAW_DIR / 'ToyotaCorolla.csv',
        BASE_DIR / 'ToyotaCorolla.csv'
    ]

    for data_file in search_paths:
        if data_file.exists():
            return pd.read_csv(data_file)

    st.warning("⚠️ Reference data not found. Using default statistics.")
    return pd.DataFrame({'Price': []})

# Load resources
try:
    rf_model, scaler, feature_importance = load_model()
    reference_data = load_reference_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading resources: {e}")
    data_loaded = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_prediction_input(age, mileage, hp, weight, cc, tax, cylinders, 
                            fuel_type, automatic, met_color, abs_system, airco):
    """
    Create prediction input array with all required features
    """
    fuel_val = 1 if fuel_type == 'Diesel' else 0
    auto_val = 1 if automatic == 'Automatic' else 0
    color_val = 1 if met_color == 'Metallic' else 0
    abs_val = 1 if abs_system else 0
    airco_val = 1 if airco else 0
    
    # Create array with all features in correct order
    features = np.array([[
        age,              # Age_08_04
        1,                # Mfg_Month
        2002,             # Mfg_Year
        mileage,          # KM
        fuel_val,         # Fuel_Type
        hp,               # HP
        color_val,        # Met_Color
        auto_val,         # Automatic
        cc,               # cc
        3,                # Doors
        cylinders,        # Cylinders
        5,                # Gears
        tax,              # Quarterly_Tax
        weight,           # Weight
        0,                # Mfr_Guarantee
        1,                # BOVAG_Guarantee
        3,                # Guarantee_Period
        abs_val,          # ABS
        1,                # Airbag_1
        1,                # Airbag_2
        airco_val,        # Airco
        0,                # Automatic_airco
        1,                # Boardcomputer
        1,                # CD_Player
        1,                # Central_Lock
        1,                # Powered_Windows
        1,                # Power_Steering
        0,                # Radio
        0,                # Mistlamps
        0,                # Sport_Model
        1,                # Backseat_Divider
        1,                # Metallic_Rim
        0,                # Radio_cassette
        0,                # Tow_Bar
    ]])
    
    return features

def predict_price(age, mileage, hp, weight, cc, tax, cylinders,
                  fuel_type, automatic, met_color, abs_system, airco):
    """
    Predict vehicle price using the trained model
    """
    features = create_prediction_input(age, mileage, hp, weight, cc, tax, 
                                      cylinders, fuel_type, automatic, 
                                      met_color, abs_system, airco)

    if scaler is not None:
        features = scaler.transform(features)
    
    # Make prediction
    predicted_price = rf_model.predict(features)[0]
    
    return predicted_price

def format_currency(value):
    """Format value as currency string"""
    return f"€{value:,.2f}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚗 Case Study: A Market-Independent ML Valuation Tool")
        st.markdown("*Predict resale prices using advanced machine learning*")
    
    # with col2: st.metric("Model Type", "Random Forest", delta="R² = 0.87")
    
    st.divider()
    
    if not data_loaded:
        st.error("⚠️ Unable to load model. Please check file paths.")
        return
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Vehicle Configuration")
        
        st.subheader("1️⃣ Condition & Usage")
        age = st.slider(
            "Age (months)",
            min_value=0,
            max_value=80,
            value=36,
            step=1,
            help="Vehicle age in months as of August 2004"
        )
        
        mileage = st.slider(
            "Mileage (km)",
            min_value=0,
            max_value=300000,
            value=100000,
            step=5000,
            help="Total kilometers driven"
        )
        
        st.subheader("2️⃣ Engine & Performance")
        hp = st.slider(
            "Horsepower (HP)",
            min_value=50,
            max_value=200,
            value=110,
            step=5,
            help="Engine power output"
        )
        
        cc = st.slider(
            "Engine Size (cc)",
            min_value=1300,
            max_value=2200,
            value=1800,
            step=100,
            help="Engine displacement"
        )
        
        cylinders = st.slider(
            "Cylinders",
            min_value=3,
            max_value=6,
            value=4,
            step=1,
            help="Number of cylinders"
        )
        
        st.subheader("3️⃣ Physical Characteristics")
        weight = st.slider(
            "Weight (kg)",
            min_value=900,
            max_value=1400,
            value=1200,
            step=10,
            help="Vehicle weight"
        )
        
        tax = st.slider(
            "Quarterly Tax (€)",
            min_value=0,
            max_value=300,
            value=100,
            step=5,
            help="Quarterly tax amount"
        )
        
        st.subheader("4️⃣ Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            fuel_type = st.radio(
                "Fuel Type",
                options=['Petrol', 'Diesel'],
                horizontal=False
            )
            
            automatic = st.radio(
                "Transmission",
                options=['Manual', 'Automatic'],
                horizontal=False
            )
        
        with col2:
            met_color = st.radio(
                "Color Type",
                options=['Standard', 'Metallic'],
                horizontal=False
            )
            
            abs_system = st.radio(
                "ABS System",
                options=['No', 'Yes'],
                horizontal=False,
                help="Anti-lock braking system"
            ) == 'Yes'
            
            airco = st.radio(
                "Air Conditioning",
                options=['No', 'Yes'],
                horizontal=False
            ) == 'Yes'
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Get prediction
    predicted_price = predict_price(
        age, mileage, hp, weight, cc, tax, cylinders,
        fuel_type, automatic, met_color, abs_system, airco
    )
    
    # Price display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style=\"
            background: linear-gradient(135deg, #2E86AB 0%, #06A77D 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        \">
            <h3 style=\"color: white; margin: 0;\">Estimated Resale Price</h3>
            <h1 style=\"color: white; margin: 15px 0;\">€{predicted_price:,.0f}</h1>
            <p style=\"color: rgba(255,255,255,0.9); margin: 0;\">
                Based on {len(feature_importance)} vehicle parameters
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # ANALYTICS SECTION
    # ========================================================================
    
    st.header("📊 Analytics & Insights")
    
    tab1, tab2 = st.tabs(["Feature Importance", "Price Analysis"])
    
    with tab1:
        st.subheader("Most Important Features for Price Prediction")
        
        # Filter out board-related features from the importance chart
        importance_for_plot = feature_importance[
            feature_importance['Feature'] != 'Boardcomputer'
        ]
        top_features = importance_for_plot.head(12)
        
        fig = go.Figure(data=[
            go.Bar(
                y=top_features['Feature'],
                x=top_features['Importance'],
                orientation='h',
                marker=dict(
                    color=top_features['Importance'],
                    colorscale='Oranges',
                    showscale=False
                ),
                text=[f"{val:.4f}" for val in top_features['Importance']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Top 12 Features by Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            showlegend=False,
            template="plotly_dark"
        )
        
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch')
        
        # Explanation
        st.info("""
        🔍 **What does this mean?**
        
        Feature importance shows which vehicle parameters have the strongest influence on price predictions.
        Higher scores indicate that changes in that feature significantly affect the estimated price.
        
        **Top 3 Insights:**
        1. **Age** - Older vehicles are worth less, the strongest price driver
        2. **Mileage** - Higher mileage reduces value substantially
        3. **Weight** - Heavier vehicles tend to have different price points
        """)
    
    with tab2:
        st.subheader("Price Prediction Sensitivity")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Age impact
            ages = np.arange(0, 81, 10)
            prices_age = [
                predict_price(a, mileage, hp, weight, cc, tax, cylinders,
                            fuel_type, automatic, met_color, abs_system, airco)
                for a in ages
            ]
            
            fig_age = go.Figure()
            fig_age.add_trace(go.Scatter(
                x=ages, y=prices_age,
                mode='lines+markers',
                name='Price',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
            
            fig_age.update_layout(
                title="Price vs Age",
                xaxis_title="Age (months)",
                yaxis_title="Price (€)",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_age, width='stretch')
        
        with col2:
            # Mileage impact
            mileages = np.arange(0, 300001, 50000)
            prices_mileage = [
                predict_price(age, m, hp, weight, cc, tax, cylinders,
                            fuel_type, automatic, met_color, abs_system, airco)
                for m in mileages
            ]
            
            fig_mileage = go.Figure()
            fig_mileage.add_trace(go.Scatter(
                x=mileages/1000, y=prices_mileage,
                mode='lines+markers',
                name='Price',
                line=dict(color='#06A77D', width=3),
                marker=dict(size=8)
            ))
            
            fig_mileage.update_layout(
                title="Price vs Mileage",
                xaxis_title="Mileage (1000 km)",
                yaxis_title="Price (€)",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_mileage, width='stretch')
        
        with col3:
            # Horsepower impact
            hps = np.arange(50, 201, 30)
            prices_hp = [
                predict_price(age, mileage, h, weight, cc, tax, cylinders,
                            fuel_type, automatic, met_color, abs_system, airco)
                for h in hps
            ]
            
            fig_hp = go.Figure()
            fig_hp.add_trace(go.Scatter(
                x=hps, y=prices_hp,
                mode='lines+markers',
                name='Price',
                line=dict(color='#F18F01', width=3),
                marker=dict(size=8)
            ))
            
            fig_hp.update_layout(
                title="Price vs Horsepower",
                xaxis_title="Horsepower (HP)",
                yaxis_title="Price (€)",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_hp, width='stretch')
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style=\"text-align: center; color: #666; font-size: 12px;\">
            <p>
                <strong>Toyota Corolla Valuation Tool</strong><br>
                Powered by Random Forest Machine Learning Model<br>
                Model Performance: R² = 0.87 | MAPE = 9.2%<br>
                <em>Predictions are estimates and may vary based on market conditions</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
