# Plotly Interactive Heatmap
fig = px.imshow(
    corr_masked,
    text_auto=".3f",
    color_continuous_scale=px.colors.sequential.Inferno_r[::-1],
    zmin=-1,
    zmax=1,
    labels=dict(x="", y="", color="corr"),
    width=1400,
    height=900,
)
fig.update_layout(
    title="Correlation Matrix of Health Features",
    xaxis_side="bottom",
    font=dict(size=14),  # Larger font
    margin=dict(l=100, r=100, t=100, b=100),
)
fig.update_xaxes(tickangle=45, tickfont=dict(size=12))
fig.update_yaxes(tickfont=dict(size=12), autorange="reversed")
st.plotly_chart(fig, use_container_width=True)

# Static Matplotlib Heatmap (for PNG download)
fig_static, ax = plt.subplots(figsize=(24, 14))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap="rocket_r",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.9},
    ax=ax,
)
ax.set_title("Correlation Matrix of Health Features", fontsize=20)