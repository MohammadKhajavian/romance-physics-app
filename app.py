# app.py - DEPLOYMENT READY VERSION
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import hashlib
from datetime import datetime
import os

# Import sklearn properly at the TOP
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Oxygen Bond Predictor", page_icon="⚛️", layout="centered")

# ------------------ CREATE DATABASE FOLDER IF NOT EXISTS ------------------
if not os.path.exists("data"):
    os.makedirs("data")

DB_PATH = "data/romance_data.db"

# ------------------ DATABASE SETUP ------------------
def init_db():
    """Create the database and table if they don't exist"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      session_id TEXT,
                      age INTEGER,
                      risk_tolerance INTEGER,
                      emotional_energy INTEGER,
                      social_orbital INTEGER,
                      first_love_duration INTEGER,
                      first_love_intensity INTEGER,
                      breakup_shock INTEGER,
                      prediction_result INTEGER,
                      predicted_time TEXT)''')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

# Initialize database
init_db()

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

# ------------------ TRAINING FUNCTION (FIXED) ------------------
def train_models():
    """Train ML models on existing data and return them"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM users", conn)
        conn.close()
        
        # If no data or less than 3 records, return dummy models
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
        
        # Features for ML
        X = df[['valence_instability', 'bond_resonance', 'recovery_entropy', 'age']]
        y_class = df['prediction_result']
        
        # Train classification model (will love happen?)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y_class)
        
        # Train regression model for time prediction (only successful cases)
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
    
    except Exception as e:
        st.warning(f"Could not train models yet. Using fallback. Error: {e}")
        return DummyClassifier(strategy="most_frequent"), DummyRegressor(strategy="mean")

# ------------------ FALLBACK PREDICTION (NO ML) ------------------
def fallback_prediction(v_inst, b_res, r_ent, age):
    """Physics-only prediction when ML not available"""
    # Physics-inspired formula
    oxygen_bond_score = (v_inst * 0.4) + (b_res * 0.4) - (r_ent * 0.2)
    
    if oxygen_bond_score > 0.45:
        # Predict 3-12 months based on age
        time_months = max(3, min(12, int(24 - (age / 5))))
        return 1, time_months
    else:
        return 0, 0

# ------------------ UI FORM ------------------
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
    st.caption("This data becomes the 'training target' for our model")
    
    love_duration = st.number_input("Duration of strongest relationship (in months)", min_value=1, max_value=240, value=24)
    intensity = st.slider("Relationship intensity (1=Low, 10=Very High)", 1, 10, 7)
    breakup_shock = st.slider("How hard was the breakup? (1=Easy, 10=Traumatic)", 1, 10, 5)
    
    submitted = st.form_submit_button("🔮 Predict My Future")

if submitted:
    # Calculate physics features for THIS user
    v_inst = calculate_valence_instability(age, risk)
    b_res = calculate_bond_resonance(love_duration, intensity)
    r_ent = calculate_recovery_entropy(breakup_shock, age)
    
    st.markdown("---")
    st.subheader("🧪 Atomic Analysis (Physics-Informed Features)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Valence Instability", f"{v_inst:.3f}", help="Younger + Risk = Higher bonding potential")
    col2.metric("🔗 Bond Resonance", f"{b_res:.3f}", help="Past bond strength (O₂ baseline)")
    col3.metric("🌪️ Recovery Entropy", f"{r_ent:.3f}", help="Breakup impact on future bonding")
    
    # Train models on ALL previous data (RETRAINS EVERY TIME)
    classification_model, regression_model = train_models()
    
    # Prepare input for prediction
    input_features = [[v_inst, b_res, r_ent, age]]
    
    # Make prediction
    try:
        will_happen = classification_model.predict(input_features)[0]
        time_months = int(regression_model.predict(input_features)[0]) if will_happen == 1 else 0
        using_ml = True
    except:
        # Fallback if ML fails
        will_happen, time_months = fallback_prediction(v_inst, b_res, r_ent, age)
        using_ml = False
    
    st.markdown("---")
    
    if will_happen == 1:
        st.success("✅ **PREDICTION: A new stable bond (O₂-like) is likely to form**")
        
        # Ensure reasonable timeframe
        time_months = max(1, min(36, time_months))
        
        st.info(f"📅 **Estimated timeframe:** {time_months} months from now")
        
        # Physics-based explanation
        if v_inst > 0.6:
            st.write("🔬 *Physics note: Your 'valence instability' is high, similar to an Oxygen atom seeking an electron partner.*")
        
        name_suggestion = "Someone with similar emotional wavelength to your past bond" if intensity >= 7 else "Someone who complements your current energy level"
        st.write(f"💡 *Hint:* {name_suggestion}")
        
        predicted_time_str = f"{time_months} months"
    else:
        st.error("❌ **PREDICTION: Low probability of a new stable bond in the near future**")
        st.write("The 'Recovery Entropy' is high relative to your 'Valence Instability' - you're in a 'Closed Shell' state.")
        predicted_time_str = "N/A"
    
    # SAVE USER DATA TO DATABASE (IMPROVES FUTURE PREDICTIONS)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT INTO users 
                     (timestamp, session_id, age, risk_tolerance, emotional_energy, social_orbital,
                      first_love_duration, first_love_intensity, breakup_shock, prediction_result, predicted_time)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                  (datetime.now().isoformat(), st.session_state['session_id'], age, risk, energy_val, social_val,
                   love_duration, intensity, breakup_shock, int(will_happen), predicted_time_str))
        conn.commit()
        conn.close()
        
        # Show model improvement message
        conn = sqlite3.connect(DB_PATH)
        new_count = pd.read_sql_query("SELECT COUNT(*) as count FROM users", conn)['count'][0]
        conn.close()
        
        st.caption(f"✓ Your data point (#{new_count}) has been added. The ML model will retrain on next prediction using all {new_count} records!")
        
    except Exception as e:
        st.warning(f"Could not save to database: {e}")

# Show current database size (proves continuous learning)
try:
    conn = sqlite3.connect(DB_PATH)
    count = pd.read_sql_query("SELECT COUNT(*) as count FROM users", conn)['count'][0]
    conn.close()
    st.markdown("---")
    st.caption(f"📊 **Continuous Learning Active:** {count} user records collected | Model retrains automatically with each prediction")
except:
    st.markdown("---")
    st.caption("📊 Database ready for first entry")
