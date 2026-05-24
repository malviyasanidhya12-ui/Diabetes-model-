import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton>button {
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stNumberInput input {
    background-color: #1f2937;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🧠 AI Health Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Smart Prediction System using Machine Learning</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
model_option = st.sidebar.selectbox("Choose Model", ["Diabetes Model"])
st.sidebar.info("Adjust inputs and get predictions instantly.")

# ---------------- MAIN LAYOUT ----------------
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    blood_pressure = st.number_input("Blood Pressure", 0, 150)

with col2:
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 1, 120)

st.divider()

# ---------------- LOAD MODEL ----------------
model = joblib.load("diabetes_model.pkl")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Now"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi,
                            dpf, age]])
    
    prediction = model.predict(input_data)[0]

    st.markdown("### 📊 Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>WELCOME</p>", unsafe_allow_html=True)