# SHAP Explainability (fixed for HTML-based force_plot)
st.markdown("#### 🧠 Model Explainability (SHAP)")
with st.expander("Show SHAP values"):
    try:
        preprocessed_input = model[:-1].transform(input_df)
        explainer = shap.Explainer(classifier, preprocessed_input)
        shap_values = explainer(preprocessed_input)

        force_html = shap.force_plot(
            explainer.expected_value[0], shap_values[0], input_df.iloc[0], matplotlib=False
        )
        st_shap(force_html)
    except Exception as e:
        st.warning(f"SHAP could not be generated: {str(e)}")
