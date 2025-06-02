import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import generate_pdf_report, st_shap, generate_recommendation
import google.generativeai as genai
import shap

def render_tabs(model, df):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧬 Recommendation Engine", "📊 Data Intelligence", 
        "🔍 Model Insights", "🤖 AI Chat Assistant", "ℹ️ About"
    ])

    with tab1:
        render_recommendation_tab(model, df)
    with tab2:
        render_data_tab(df)
    with tab3:
        render_model_tab(df)
    with tab4:
        render_chat_tab()
    with tab5:
        render_about_tab()

def render_recommendation_tab(model, df):
    st.subheader("Generate Personalized Recommendation")
    frequency = st.sidebar.slider("📅 Visit Frequency", 0, 50, 5)
    monetary = st.sidebar.slider("💸 Annual Spending ($)", 0, 10000, 500)
    time = st.sidebar.slider("⏳ Time Since Last Visit", 0, 60, 12)

    if st.sidebar.button("💡 Predict Risk & Recommend"):
        input_df = pd.DataFrame({"Frequency": [frequency], "Monetary": [monetary], "Time": [time]})
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = f"{probs[prediction]*100:.2f}%"
        recommendation = generate_recommendation(prediction)

        summary = f"""Risk: {['Low', 'Medium', 'High'][prediction]}\nConfidence: {confidence}\nVisit: {frequency}\nSpend: ${monetary}\nLast Visit: {time} months\n\nRecommendation: {recommendation}"""
        st.session_state["health_summary"] = summary

        with st.spinner("Analyzing with Gemini..."):
            chat = genai.GenerativeModel("gemini-1.5-flash-latest").start_chat()
            prompt = f"{summary}\n\nSuggest likely causes, treatments, and specialists."
            ai_response = chat.send_message(prompt)
            st.markdown("### 🧠 Gemini's Analysis")
            st.markdown(ai_response.text)

            path = generate_pdf_report(summary, ai_response.text)
            with open(path, "rb") as f:
                st.download_button("📄 Download PDF", f, file_name="health_report.pdf", mime="application/pdf")

        show_visuals(model, input_df, df, prediction, confidence, recommendation)

def show_visuals(model, input_df, df, prediction, confidence, recommendation):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Level", ["Low", "Medium", "High"][prediction])
        st.metric("Confidence", confidence)
        st.markdown(recommendation)

        try:
            clf_step = next(s for s in model.named_steps if hasattr(model.named_steps[s], "coef_"))
            importance = model.named_steps[clf_step].coef_[0]
        except Exception:
            importance = [0, 0, 0]
        st.plotly_chart(px.bar(x=["Frequency", "Monetary", "Time"], y=importance, title="Feature Weights"))

        st.markdown("#### 🧠 SHAP Explainability")
        try:
            preprocessor = model.named_steps["preprocessor"]
            transformed = preprocessor.transform(input_df)
            explainer = shap.Explainer(model.named_steps[clf_step])
            shap_vals = explainer(transformed)
            st_shap(shap.plots.waterfall(shap_vals[0]))
        except Exception as e:
            st.warning(f"SHAP not available: {e}")

    with col2:
        avg = [df[col].mean() for col in ["Frequency", "Monetary", "Time"]]
        st.plotly_chart(go.Figure([
            go.Bar(x=["Frequency", "Monetary", "Time"], y=input_df.values.flatten(), name="You"),
            go.Scatter(x=["Frequency", "Monetary", "Time"], y=avg, name="Average", mode="lines+markers")
        ]).update_layout(title="You vs Population"), use_container_width=True)

def render_data_tab(df):
    st.subheader("📊 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
        st.dataframe(df.describe())
    with col2:
        st.plotly_chart(px.pie(df, names="Class", title="Risk Distribution", hole=0.3))
        st.dataframe(df.isnull().sum().to_frame("Missing"))

def render_model_tab(df):
    st.subheader("📈 Model Interpretability")
    st.plotly_chart(px.imshow(df.corr(numeric_only=True), title="Correlations", color_continuous_scale="RdBu"))
    feat = st.selectbox("View by Feature", ["Frequency", "Monetary", "Time"])
    st.plotly_chart(px.box(df, x="Class", y=feat, color="Class", title=f"{feat} by Risk"))

def render_chat_tab():
    st.subheader("🤖 Chat with AI")
    query = st.text_area("💬 Ask a health-related question:")
    if st.button("🚀 Submit"):
        context = st.session_state.get("health_summary", "")
        prompt = f"{context}\n\nQuery: {query}"
        try:
            chat = genai.GenerativeModel("gemini-1.5-flash-latest").start_chat()
            response = chat.send_message(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"AI Chat Error: {e}")

def render_about_tab():
    st.subheader("ℹ️ About")
    st.markdown("""
    An intelligent healthcare assistant powered by:
    - Logistic Regression & SHAP
    - Streamlit, Plotly, and FPDF
    - Google Gemini AI for natural language insights

    **Developer:** Jayanth  
    🔗 [GitHub](https://github.com/Jayanth2323/HealthCare)
    """)
