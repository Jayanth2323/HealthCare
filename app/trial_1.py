# === SHAP Explainability (dynamic step detection) ===
st.markdown("#### 🧠 Model Explainability (SHAP)")
with st.expander("Show SHAP values"):
    try:
        # ➤ Debug: Show model pipeline steps
        st.write("🔍 Model pipeline steps:", model.named_steps)

        preprocessor = None
        classifier = None

        # ➤ Scan top-level and nested pipeline steps
        for name, step in model.named_steps.items():
            if isinstance(step, Pipeline):
                for sub_name, sub_step in step.named_steps.items():
                    if hasattr(sub_step, "transform") and not hasattr(sub_step, "predict_proba"):
                        preprocessor = sub_step
                    if hasattr(sub_step, "predict_proba"):
                        classifier = sub_step
            else:
                if hasattr(step, "transform") and not hasattr(step, "predict_proba"):
                    preprocessor = step
                if hasattr(step, "predict_proba"):
                    classifier = step

        # ➤ Validate detection
        if preprocessor is None:
            raise ValueError("No transformer found in pipeline.")
        if classifier is None:
            raise ValueError("No classifier found in pipeline.")

        # ➤ Transform input data
        X_pre = preprocessor.transform(input_df)

        # ➤ Use SHAP with KernelExplainer
        explainer = shap.KernelExplainer(
            classifier.predict_proba,
            shap.sample(X_pre, 10)
        )
        shap_vals = explainer.shap_values(X_pre, nsamples=100)

        # ➤ Handle binary vs multiclass
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            class_idx = 1  # For multiclass/high-risk class
        else:
            class_idx = 0  # For binary

        # ➤ Plot SHAP summary
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_vals[class_idx] if isinstance(shap_vals, list) else shap_vals,
            features=X_pre,
            feature_names=input_df.columns.tolist(),
            plot_type="bar",
            show=False,
            alpha=0.8
        )
        ax.set_title("SHAP Feature Contribution")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ SHAP could not be generated: {e}")