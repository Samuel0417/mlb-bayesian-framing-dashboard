import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------
# Page Config
# ------------------------------------------------

st.set_page_config(
    page_title="MLB Hybrid Framing Dashboard",
    layout="wide"
)

# ------------------------------------------------
# Load Data
# ------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("hybrid_dashboard_grid_portfolio_small.csv")

df = load_data()

# ------------------------------------------------
# Basic Cleaning
# ------------------------------------------------

df["catcher_name"] = df["catcher_name"].fillna(df["catcher"].astype(str))

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

st.sidebar.title("Controls")

selected_catcher = st.sidebar.selectbox(
    "Catcher",
    sorted(df["catcher_name"].unique())
)

pitch_type = st.sidebar.selectbox(
    "Pitch Type",
    sorted(df["pitch_type"].unique())
)

balls = st.sidebar.selectbox(
    "Balls",
    sorted(df["balls"].unique())
)

strikes = st.sidebar.selectbox(
    "Strikes",
    sorted(df["strikes"].unique())
)

stand = st.sidebar.selectbox(
    "Batter Stand",
    sorted(df["stand"].unique())
)

p_throws = st.sidebar.selectbox(
    "Pitcher Throws",
    sorted(df["p_throws"].unique())
)

heatmap_type = st.sidebar.radio(
    "Heatmap Type",
    [
        "Adjusted Strike Probability",
        "Framing Effect",
        "LightGBM Baseline Probability"
    ]
)

# ------------------------------------------------
# Filter Data
# ------------------------------------------------

plot_df = df[
    (df["catcher_name"] == selected_catcher) &
    (df["pitch_type"] == pitch_type) &
    (df["balls"] == balls) &
    (df["strikes"] == strikes) &
    (df["stand"] == stand) &
    (df["p_throws"] == p_throws)
].copy()

if plot_df.empty:
    st.warning("No data available for this combination.")
    st.stop()

# ------------------------------------------------
# Display Column
# ------------------------------------------------

if heatmap_type == "Adjusted Strike Probability":
    color_col = "adjusted_prob"
    color_title = "Adjusted Strike Probability"
    zmin, zmax = 0, 1

elif heatmap_type == "Framing Effect":
    color_col = "framing_effect"
    color_title = "Bayesian Framing Effect"
    zmin, zmax = -0.25, 0.25

else:
    color_col = "lightgbm_prob"
    color_title = "LightGBM Baseline Probability"
    zmin, zmax = 0, 1

# ------------------------------------------------
# Title / Metrics
# ------------------------------------------------

st.title("⚾ MLB Hybrid Catcher Framing Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Catcher", selected_catcher)

with col2:
    st.metric("Count", f"{balls}-{strikes}")

with col3:
    st.metric("Matchup", f"{stand} batter vs {p_throws} pitcher")

# ------------------------------------------------
# Data for Heatmap
# ------------------------------------------------

heatmap_df = plot_df.pivot_table(
    index="plate_z",
    columns="plate_x",
    values=color_col,
    aggfunc="mean"
).sort_index()

x_vals = heatmap_df.columns.values
y_vals = heatmap_df.index.values
z_vals = heatmap_df.values

# ------------------------------------------------
# Heatmap
# ------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale="RdBu_r",
        zmin=zmin,
        zmax=zmax,
        zsmooth="best",
        colorbar=dict(
            title=color_title
        ),
        hovertemplate=
            "Plate X=%{x:.2f}<br>" +
            "Plate Z=%{y:.2f}<br>" +
            f"{color_title}=%{{z:.3f}}<extra></extra>"
    )
)

# Strike zone box
fig.add_shape(
    type="rect",
    x0=-0.83,
    x1=0.83,
    y0=1.5,
    y1=3.5,
    line=dict(color="black", width=3),
    fillcolor="rgba(0,0,0,0)"
)

fig.update_layout(
    template="plotly_dark",
    height=780,
    font=dict(size=16),
    title=dict(
        text=(
            f"{selected_catcher} | {pitch_type} | "
            f"Count {balls}-{strikes} | "
            f"{stand} batter vs {p_throws} pitcher"
        ),
        font=dict(size=22)
    )
)

fig.update_xaxes(
    title="Plate X",
    range=[-2, 2]
)

fig.update_yaxes(
    title="Plate Z",
    range=[0, 5],
    scaleanchor="x",
    scaleratio=1
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Details
# ------------------------------------------------

with st.expander("Model Prediction Summary"):
    summary_df = pd.DataFrame({
        "Metric": [
            "Average LightGBM Baseline Probability",
            "Average Adjusted Strike Probability",
            "Average Bayesian Framing Effect",
            "Maximum Framing Gain",
            "Maximum Framing Loss"
        ],
        "Value": [
            plot_df["lightgbm_prob"].mean(),
            plot_df["adjusted_prob"].mean(),
            plot_df["framing_effect"].mean(),
            plot_df["framing_effect"].max(),
            plot_df["framing_effect"].min()
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

# ------------------------------------------------
# Footer
# ------------------------------------------------

st.markdown("---")

st.markdown("""
### Model Information

This dashboard uses a hybrid sports analytics framework:

- **LightGBM baseline model** estimates called-strike probability using full pitch-level data.
- **Bayesian hierarchical framing model** estimates catcher-specific spatial framing effects.
- **Adjusted strike probability** combines the LightGBM baseline with the Bayesian framing effect.

### Final Model Performance

| Model | Test AUC | Test Log Loss |
|---|---:|---:|
| LightGBM Baseline | 0.9837 | 0.1546 |
| Bayesian + Handedness | 0.9804 | 0.1668 |
| CNN | 0.9811 | 0.1665 |
""")
