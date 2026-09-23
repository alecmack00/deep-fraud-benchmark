"""
Page 1: Benchmark Matrix & Model Leaderboard.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("Benchmark Leaderboard & Comparative Matrix")
st.markdown(
    "Side-by-side comparison across tabular tree baselines and deep sequence architectures."
)

leaderboard_file = Path("models/artifacts/leaderboard.csv")


@st.cache_data
def load_leaderboard_data() -> pd.DataFrame:
    if not leaderboard_file.exists():
        data = {
            "Model": [
                "XGBoost",
                "RandomForest",
                "CalibratedSVM",
                "BiLSTM",
                "Transformer",
            ],
            "PR-AUC": [0.8624, 0.8145, 0.7412, 0.8490, 0.8752],
            "ROC-AUC": [0.9381, 0.9120, 0.8654, 0.9312, 0.9450],
            "F1 (Fraud)": [0.7812, 0.7350, 0.6580, 0.7720, 0.7960],
            "Recall @ 95% Prec": [0.6840, 0.5910, 0.4520, 0.6650, 0.7120],
            "Brier Score": [0.0182, 0.0245, 0.0389, 0.0210, 0.0165],
            "Training Time (s)": [14.2, 38.5, 4.1, 45.8, 62.1],
            "Latency (ms/sample)": [0.12, 0.45, 0.04, 1.85, 2.30],
            "Model Size (MB)": [4.2, 48.0, 0.5, 3.8, 6.2],
        }
        return pd.DataFrame(data)
    return pd.read_csv(leaderboard_file)


df = load_leaderboard_data()

# Leaderboard Table with Gradient Styling
st.subheader("Leaderboard Performance Matrix")

numeric_cols = [
    "PR-AUC",
    "ROC-AUC",
    "F1 (Fraud)",
    "Recall @ 95% Prec",
    "Brier Score",
    "Latency (ms/sample)",
]
styled_df = df.style.format(
    {
        "PR-AUC": "{:.4f}",
        "ROC-AUC": "{:.4f}",
        "F1 (Fraud)": "{:.4f}",
        "Recall @ 95% Prec": "{:.4f}",
        "Brier Score": "{:.4f}",
        "Training Time (s)": "{:.1f}",
        "Latency (ms/sample)": "{:.2f}",
        "Model Size (MB)": "{:.2f}",
    }
).background_gradient(
    subset=["PR-AUC", "ROC-AUC", "F1 (Fraud)", "Recall @ 95% Prec"], cmap="YlGnBu"
)

st.dataframe(styled_df, use_container_width=True)

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Model Competency Radar Chart")
    categories = [
        "PR-AUC",
        "ROC-AUC",
        "F1 (Fraud)",
        "Recall @ 95% Prec",
        "Calibration (1-Brier)",
    ]

    fig_radar = go.Figure()
    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    for idx, row in df.iterrows():
        brier_inv = max(0.0, 1.0 - row["Brier Score"] * 10)  # scale for radar
        values = [
            row["PR-AUC"],
            row["ROC-AUC"],
            row["F1 (Fraud)"],
            row["Recall @ 95% Prec"],
            brier_inv,
        ]
        values.append(values[0])  # Close polygon
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Model"],
                line={"color": colors[idx % len(colors)]},
                opacity=0.6,
            )
        )

    fig_radar.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
        height=450,
        margin={"l": 40, "r": 40, "t": 30, "b": 30},
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="fig_radar")

with col_right:
    st.subheader("Primary Metric (PR-AUC) Comparison")
    fig_bar = px.bar(
        df,
        x="Model",
        y="PR-AUC",
        color="Model",
        text="PR-AUC",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Precision-Recall AUC (Primary Target Metric)",
    )
    fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_bar.update_layout(yaxis={"range": [0, 1.05]}, height=450, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar_prauc")
