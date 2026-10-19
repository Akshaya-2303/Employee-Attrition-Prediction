"""Streamlit web interface.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import load_model, predict_attrition  # noqa: E402
from utils import DEPARTMENTS  # noqa: E402

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥")
st.title("👥 Employee Attrition Predictor")
st.write("Enter employee details to estimate whether they are likely to leave.")


@st.cache_resource
def get_model():
    return load_model()


try:
    bundle = get_model()
except (FileNotFoundError, RuntimeError) as err:
    st.error(str(err))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 70, 30)
    department = st.selectbox("Department", DEPARTMENTS)
    income = st.number_input("Monthly income (₹)", 1000, 500000, 50000, step=1000)
    years = st.number_input("Years at company", 0.0, 45.0, 3.0, step=0.5)
    distance = st.number_input("Distance from home (km)", 0, 100, 10)
with col2:
    job_sat = st.slider("Job satisfaction", 1, 5, 3)
    work_life = st.slider("Work-life balance", 1, 5, 3)
    performance = st.slider("Performance rating", 1, 5, 3)
    overtime = st.radio("Works overtime?", ["No", "Yes"], horizontal=True)

if st.button("Predict", type="primary"):
    try:
        result = predict_attrition({
            "Age": age, "Department": department, "MonthlyIncome": income,
            "YearsAtCompany": years, "JobSatisfaction": job_sat,
            "WorkLifeBalance": work_life, "PerformanceRating": performance,
            "OverTime": overtime, "DistanceFromHome": distance,
        }, bundle)
        prob = result["probability"]
        if result["prediction"] == "Yes":
            st.error(f"⚠️ Likely to LEAVE — probability {prob:.1%}")
        else:
            st.success(f"✅ Likely to STAY — probability of leaving {prob:.1%}")
        st.progress(prob)
    except ValueError as err:
        st.warning(f"Invalid input: {err}")
