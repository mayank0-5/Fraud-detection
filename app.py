# ============================================
# app.py - Streamlit Web App
# Credit Card Fraud Detection
# ============================================

import streamlit as st
import numpy as np
import joblib

# ── Load Model ──────────────────────────────
model = joblib.load("model.pkl")

# ── App Title ───────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below to check if it is fraudulent.")

# ── Input Fields ────────────────────────────
st.subheader("Transaction Details")

amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=100.0)
time   = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=50000.0)

st.write("Enter V1 to V28 features (from PCA):")

cols = st.columns(4)
v = []
for i in range(1, 29):
    with cols[(i-1) % 4]:
        val = st.number_input(f"V{i}", value=0.0, format="%.4f")
        v.append(val)

# ── Predict Button ───────────────────────────
if st.button("🔍 Check Transaction"):
    amount_scaled = (amount - 88.35) / 250.12
    time_scaled   = (time - 94813.86) / 47488.14

    features = np.array(v + [amount_scaled, time_scaled]).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED! Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ LEGITIMATE Transaction! Fraud Probability: {probability*100:.2f}%")