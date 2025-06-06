# PCA Visualization
st.markdown("#### 🧬 PCA: Patient Clusters")
numeric_df = df.select_dtypes(include="number").dropna()

# Standardize and run PCA on all numeric columns
scaler = StandardScaler()
scaled = scaler.fit_transform(numeric_df)
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)
reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])

# Only plot if "diabetes" exists
if "diabetes" in df.columns:
    # Add the diabetes label back onto reduced_df
    reduced_df["diabetes"] = df.loc[numeric_df.index, "diabetes"].values

    fig_pca = px.scatter(
        reduced_df,
        x="PC1",
        y="PC2",
        color="diabetes",
        title="PCA Projection of Patient Features"
    )
    st.plotly_chart(fig_pca, use_container_width=True)
else:
    st.info("⚠️ Cannot show PCA plot because the 'diabetes' column is missing.")
