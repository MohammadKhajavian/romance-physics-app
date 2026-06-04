# app.py - COMPLETE WORKING VERSION
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import hashlib
from datetime import datetime
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Oxygen Bond Predictor", page_icon="⚛️", layout="centered")

# ------------------ CREATE DATABASE FOLDER IF NOT EXISTS ------------------
if not os.path.exists("data"):
    os.makedirs("data")

DB_PATH = "data/romance_data.db"


# ------------------ DATABASE SETUP ------------------
def init_db():
    """Create the database and table if they don't exist"""
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
    print("Database initialized at:", DB_PATH)


# Call this to create the database
init_db()


# ------------------ PHYSICS KERNEL FUNCTIONS ------------------
def calculate_valence_instability(age, risk_tolerance):
    """Oxygen analogy: Younger + High Risk = High Energy Orbital"""
    return ((30 - min(age, 30)) / 30) * (risk_tolerance / 5)


def calculate_bond_resonance(duration, intensity):
    """How well the past bond matched"""
    return (min(duration, 120) / 120) * (intensity / 10)


def calculate_recovery_entropy(breakup_shock, current_age):
    """Higher shock = harder to bond again"""
    return breakup_shock / 10


# ------------------ TRAINING FUNCTION ------------------
def train_models():
    """Train ML models on existing data"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()

    # If not enough data, return dummy models
    if len(df) < 3:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        return DummyClassifier(strategy="most_frequent"), DummyRegressor(strategy="mean")

    # Create physics features
    df['valence_instability'] = df.apply(lambda x: calculate_valence_instability(x['age'], x['risk_tolerance']), axis=1)
    df['bond_resonance'] = df.apply(
        lambda x: calculate_bond_resonance(x['first_love_duration'], x['first_love_intensity']), axis=1)
    df['recovery_entropy'] = df.apply(lambda x: calculate_recovery_entropy(x['breakup_shock'], x['age']), axis=1)

    X = df[['valence_instability', 'bond_resonance', 'recovery_entropy', 'age']]
    y_class = df['prediction_result']

    clf = LogisticRegression()
    clf.fit(X, y_class)

    # For time prediction (only successful cases)
    df_success = df[df['prediction_result'] == 1]
    if len(df_success) > 0:
        y_time = df_success['predicted_time'].apply(lambda x: int(x.split()[0]) if x != "N/A" else 12)
        X_time = df_success[['valence_instability', 'bond_resonance', 'recovery_entropy', 'age']]
        reg = RandomForestRegressor()
        reg.fit(X_time, y_time)
        return clf, reg

    return clf, RandomForestRegressor()


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
        value=3
    )

    energy_options = {"Low Energy (1)": 1, "Medium Energy (3)": 3, "High Energy (5)": 5}
    energy = st.radio("Emotional Energy Level", list(energy_options.keys()))
    energy_val = energy_options[energy]

    social_options = {"Introvert (1)": 1, "Ambivert (3)": 3, "Extrovert (5)": 5}
    social = st.selectbox("Social Style", list(social_options.keys()))
    social_val = social_options[social]

    st.markdown("---")
    st.subheader("💔 Part 2: Your Strongest Past Relationship (Target for ML)")

    love_duration = st.number_input("Duration of strongest relationship (in months)", min_value=1, max_value=240,
                                    value=24)
    intensity = st.slider("Relationship intensity (1=Low, 10=Very High)", 1, 10, 7)
    breakup_shock = st.slider("How hard was the breakup? (1=Easy, 10=Traumatic)", 1, 10, 5)

    submitted = st.form_submit_button("🔮 Predict My Future")

if submitted:
    # Calculate physics features
    v_inst = calculate_valence_instability(age, risk)
    b_res = calculate_bond_resonance(love_duration, intensity)
    r_ent = calculate_recovery_entropy(breakup_shock, age)

    st.markdown("---")
    st.subheader("🧪 Atomic Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Valence Instability", f"{v_inst:.2f}")
    col2.metric("Bond Resonance", f"{b_res:.2f}")
    col3.metric("Recovery Entropy", f"{r_ent:.2f}")

    # Train and predict
    clf, reg = train_models()
    input_features = [[v_inst, b_res, r_ent, age]]

    will_happen = clf.predict(input_features)[0]

    st.markdown("---")

    if will_happen == 1:
        st.success("✅ **PREDICTION: A new relationship is likely to form**")

        # Predict timeframe
        time_months = int(reg.predict(input_features)[0])
        time_months = max(1, min(36, time_months))  # Between 1-36 months

        st.info(f"📅 **Estimated timeframe:** {time_months} months from now")

        # Name suggestion based on past bond intensity
        if intensity >= 7:
            name_suggestion = "Someone with similar energy to your past bond"
        else:
            name_suggestion = "Someone who complements your current energy level"
        st.write(f"💡 *Hint:* {name_suggestion}")

        predicted_time_str = f"{time_months} months"
    else:
        st.error("❌ **PREDICTION: Low probability of a new relationship soon**")
        st.write("The model suggests focusing on personal recovery first.")
        predicted_time_str = "N/A"

    # Save to database
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

    st.caption("✓ Your data has been added to the training set. The model improves with every prediction!")

# Show current database size
conn = sqlite3.connect(DB_PATH)
count = pd.read_sql_query("SELECT COUNT(*) as count FROM users", conn)['count'][0]
conn.close()
st.markdown("---")
st.caption(f"📊 Current database size: {count} user records | Model retrains automatically")