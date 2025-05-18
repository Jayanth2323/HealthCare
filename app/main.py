import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/logistic_regression_pipeline.pkl")


# Recommendation logic
def generate_recommendation(prediction_label):
    recommendations = {
        "low_risk": """Maintain current lifestyle.
        Regular annual check-ups recommended.""",
        "medium_risk": """Increase physical activity and monitor diet.
        Schedule a medical consultation.""",
        "high_risk": """Immediate medical attention advised.
        Begin treatment under supervision.""",
    }
    return recommendations.get(
        str(prediction_label), "No recommendation available.")


# Streamlit App
st.set_page_config(
    page_title="Healthcare Recommendation System", layout="centered")
st.title("🩺 Personalized Healthcare Recommendation System")

# Inputs based on your trained model
frequency = st.slider("Visit Frequency (visits per year)", 0, 50, 5)
monetary = st.slider("Healthcare Spending ($)", 0, 10000, 500)
time = st.slider("Time Since Last Visit (months)", 0, 60, 12)

if st.button("Get Recommendation"):
    input_data = pd.DataFrame(
        {
            "Frequency": [frequency],
            "Monetary": [monetary],
            "Time": [time],
        }
    )

    prediction = model.predict(input_data)[0]
    recommendation = generate_recommendation(prediction)

    st.markdown(f"### 🧠 Predicted Health Risk: **{prediction}**")
    st.success(f"✅ Recommended Action: {recommendation}")
