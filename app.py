# app.py - COMPLETE WORKING VERSION WITH FIXED SUPABASE
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import os

# Import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

# Supabase import
from supabase import create_client, Client

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Oxygen Bond Predictor", page_icon="⚛️", layout="centered")

# ------------------ SUPABASE SETUP (Permanent Storage) ------------------
@st.cache_resource
def init_supabase():
    """Initialize connection to Supabase (cloud database)"""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        # Remove any trailing slashes from URL
        url = url.rstrip('/')
        # Remove /rest/v1/ if accidentally included
        if url.endswith('/rest/v1'):
            url = url.replace('/rest/v1', '')
        return create_client(url, key)
    except Exception as e:
        st.error(f"Cannot connect to Supabase. Please check your secrets. Error: {e}")
        return None

# Initialize Supabase
supabase = init_supabase()

# ------------------ PHYSICS KERNEL FUNCTIONS ------------------
def calculate_valence_instability(age, risk_tolerance):
    """Oxygen analogy: Younger + High Risk = High Energy Orbital"""
    try:
        age_factor = (30 - min(age, 30)) / 30
        risk_factor = risk_tolerance / 5
        return round(age_factor * risk_factor, 3)
    except:
        return 0.5

def calculate_bond_resonance(duration, intensity):
    """How well the past bond matched - O2 bond strength analogy"""
    try:
        duration_factor = min(duration, 120) / 120
        intensity_factor = intensity / 10
        return round(duration_factor * intensity_factor, 3)
    except:
        return 0.5

def calculate_recovery_entropy(breakup_shock, current_age):
    """Higher shock = harder to bond again (Entropy increase)"""
    try:
        return round(breakup_shock / 10, 3)
    except:
        return 0.5

# ------------------ FETCH DATA FROM SUPABASE ------------------
def fetch_all_data():
    """Get all user data from Supabase cloud database"""
    if supabase is None:
        return pd.DataFrame()
    try:
        response = supabase.table("users").select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not fetch data: {e}")
        return pd.DataFrame()

# ------------------ TRAINING FUNCTION ------------------
def train_models():
    """Train ML models on ALL existing data from Supabase"""
    df = fetch_all_data()
    
    if len(df) < 3:
        return DummyClassifier(strategy="most_frequent"), DummyRegressor(strategy="mean")
    
    # Create physics-informed features
    df['valence_instability'] = df.apply(
        lambda x: calculate_valence_instability(x['age'], x['risk_tolerance']), axis=1
    )
    df['bond_resonance'] = df.apply(
        lambda x: calculate_bond_resonance(x['first_love_duration'], x['first_love_intensity']), axis=1
    )
    df['recovery_entropy'] = df.apply(
        lambda x: calculate_recovery_entropy(x['breakup_shock'], x['age']), axis=1
    )
    
    X = df[['valence_instability', 'bond_resonance', 'recovery_entropy', 'age']]
    y_class = df['prediction_result']
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y_class)
    
    df_success = df[df['prediction_result'] == 1]
    if len(df_success) > 0:
        y_time = df_success['predicted_time'].apply(
            lambda x: int(x.split()[0]) if x != "N/A" and x != "0 months" else 6
        )
        X_time = df_success[['valence_instability', 'bond_resonance', 'recovery_entropy', 'age']]
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        reg.fit(X_time, y_time)
        return clf, reg
    
    return clf, DummyRegressor(strategy="mean")

# ------------------ SAVE DATA TO SUPABASE ------------------
def save_to_supabase(data):
    """Save user prediction to cloud database"""
    if supabase is None:
        st.warning("Cannot save: No database connection")
        return False
    try:
        result = supabase.table("users").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return False

# ------------------ UI (FRONTEND) ------------------
st.title("⚛️ Quantum Romance Predictor")
st.caption("Based on Atomic Interaction Modelling (O₂ Bond Analogy)")
st.markdown("---")

# Session ID for tracking
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = hashlib.md5(str(datetime.now()).encode()).hexdigest()

with st.form("prediction_form"):
    st.subheader("📡 Part 1: Your Personality (Features for ML)")
    
    age = st.number_input("Your Age", min_value=18, max_value=100, value=25, step=1)
    
    risk = st.select_slider(
        "Risk Tolerance (1=Very Cautious, 5=Very Adventurous)",
        options=[1, 2, 3, 4, 5],
        value=3,
        help="Oxygen analogy: Higher risk = Higher orbital energy = More likely to bond"
    )
    
    energy_options = {"Low Energy (1)": 1, "Medium Energy (3)": 3, "High Energy (5)": 5}
    energy = st.radio("Emotional Energy Level", list(energy_options.keys()))
    energy_val = energy_options[energy]
    
    social_options = {"Introvert (1)": 1, "Ambivert (3)": 3, "Extrovert (5)": 5}
    social = st.selectbox("Social Style", list(social_options.keys()))
    social_val = social_options[social]
    
    st.markdown("---")
    st.subheader("💔 Part 2: Your Strongest Past Relationship (Target for ML)")
    
    love_duration = st.number_input("Duration of strongest relationship (in months)", min_value=1, max_value=240, value=24)
    intensity = st.slider("Relationship intensity (1=Low, 10=Very High)", 1, 10, 7)
    breakup_shock = st.slider("How hard was the breakup? (1=Easy, 10=Traumatic)", 1, 10, 5)
    
    submitted = st.form_submit_button("🔮 Predict My Future")

if submitted:
    # Calculate physics features
    v_inst = calculate_valence_instability(age, risk)
    b_res = calculate_bond_resonance(love_duration, intensity)
    r_ent = calculate_recovery_entropy(breakup_shock, age)
    
    st.markdown("---")
    st.subheader("🧪 Atomic Analysis (Physics-Informed Features)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Valence Instability", f"{v_inst:.3f}")
    col2.metric("🔗 Bond Resonance", f"{b_res:.3f}")
    col3.metric("🌪️ Recovery Entropy", f"{r_ent:.3f}")
    
    # Train models on ALL previous data
    classification_model, regression_model = train_models()
    input_features = [[v_inst, b_res, r_ent, age]]
    
    try:
        will_happen = classification_model.predict(input_features)[0]
        time_months = int(regression_model.predict(input_features)[0]) if will_happen == 1 else 0
    except:
        # Fallback prediction
        oxygen_bond_score = (v_inst * 0.4) + (b_res * 0.4) - (r_ent * 0.2)
        will_happen = 1 if oxygen_bond_score > 0.45 else 0
        time_months = max(3, min(12, int(24 - (age / 5)))) if will_happen == 1 else 0
    
    st.markdown("---")
    
    if will_happen == 1:
        st.success("✅ **PREDICTION: A new relationship is likely to form**")
        time_months = max(1, min(36, time_months))
        st.info(f"📅 **Estimated timeframe:** {time_months} months from now")
        predicted_time_str = f"{time_months} months"
    else:
        st.error("❌ **PREDICTION: Low probability of a new relationship soon**")
        predicted_time_str = "N/A"
    
    # SAVE TO CLOUD DATABASE (PERMANENT)
    user_data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state['session_id'],
        "age": age,
        "risk_tolerance": risk,
        "emotional_energy": energy_val,
        "social_orbital": social_val,
        "first_love_duration": love_duration,
        "first_love_intensity": intensity,
        "breakup_shock": breakup_shock,
        "prediction_result": int(will_happen),
        "predicted_time": predicted_time_str
    }
    
    if save_to_supabase(user_data):
        # Get total count to show learning progress
        df_all = fetch_all_data()
        st.caption(f"✓ Data saved to cloud! Total {len(df_all)} users. Model improves with each new user!")
    else:
        st.warning("Data could not be saved. Please try again.")

# Show current database size
df_all = fetch_all_data()
st.markdown("---")
st.caption(f"📊 **Continuous Learning Active:** {len(df_all)} total user records | Model retrains automatically")
