"""
Page 3: Latent Space Explorer (3D PCA & K-Means Cluster Visualizer).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.title("Latent Space Explorer: 3D PCA & K-Means Clusters")
st.markdown(
    "Explore unsupervised transaction embeddings in orthogonal PCA latent space. "
    "Visualize behavioral spending clusters and inspect how fraudulent activities separate from legitimate behavior."
)


@st.cache_data
def load_latent_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    latent_path = Path("models/artifacts/latent_space_dev.npz")
    if latent_path.exists():
        data = np.load(latent_path)
        coords = data["coords"][:, :3]
        labels = data["labels"]
        clusters = data["clusters"]
        centroids = (
            data["centroids"][:, :3]
            if data["centroids"].shape[1] >= 3
            else data["centroids"]
        )
        scree_var = data["scree_var"]
        cum_var = data["cum_var"]
        return coords, labels, clusters, centroids, scree_var, cum_var

    np.random.seed(42)
    n_pts = 1500
    clusters = np.random.randint(0, 8, size=n_pts)
    centroids = np.random.uniform(-3, 3, size=(8, 3))
    coords = centroids[clusters] + np.random.normal(0, 0.6, size=(n_pts, 3))
    labels = (np.random.uniform(0, 1, size=n_pts) < 0.04).astype(int)
    coords[labels == 1] += np.random.normal(
        1.5, 0.8, size=(int(np.sum(labels == 1)), 3)
    )
    scree_var = np.array([0.28, 0.19, 0.14, 0.09, 0.07, 0.05, 0.04, 0.03])
    cum_var = np.cumsum(scree_var)
    return coords, labels, clusters, centroids, scree_var, cum_var


coords, labels, clusters, centroids, scree_var, cum_var = load_latent_data()

# UI Controls
col_ctrl1, col_ctrl2 = st.columns([1, 2])
with col_ctrl1:
    color_by = st.selectbox(
        "Color Data Points By:", ["Fraud Flag (isFraud)", "K-Means Cluster ID"]
    )
with col_ctrl2:
    highlight_fraud = st.checkbox("Highlight Fraud Outliers in Red Glow", value=True)

# Build 3D Plotly Scatter
df_plot = pd.DataFrame(
    {
        "PCA_1": coords[:, 0],
        "PCA_2": coords[:, 1],
        "PCA_3": coords[:, 2],
        "isFraud": ["Fraud (1)" if y == 1 else "Legitimate (0)" for y in labels],
        "Cluster": [f"Cluster {c}" for c in clusters],
    }
)

fig_3d = go.Figure()

if color_by == "Fraud Flag (isFraud)":
    for label_val, color_code in [
        ("Legitimate (0)", "#3b82f6"),
        ("Fraud (1)", "#ef4444"),
    ]:
        subset = df_plot[df_plot["isFraud"] == label_val]
        marker_size = 6 if label_val == "Fraud (1)" and highlight_fraud else 3
        opacity = 0.9 if label_val == "Fraud (1)" else 0.5
        fig_3d.add_trace(
            go.Scatter3d(
                x=subset["PCA_1"],
                y=subset["PCA_2"],
                z=subset["PCA_3"],
                mode="markers",
                name=label_val,
                marker={"size": marker_size, "color": color_code, "opacity": opacity},
            )
        )
else:
    for cluster_id in sorted(df_plot["Cluster"].unique()):
        subset = df_plot[df_plot["Cluster"] == cluster_id]
        fig_3d.add_trace(
            go.Scatter3d(
                x=subset["PCA_1"],
                y=subset["PCA_2"],
                z=subset["PCA_3"],
                mode="markers",
                name=cluster_id,
                marker={"size": 3, "opacity": 0.6},
            )
        )

# Add K-Means Centroids
fig_3d.add_trace(
    go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode="markers+text",
        name="Centroids",
        marker={
            "size": 9,
            "color": "#f59e0b",
            "symbol": "diamond",
            "line": {"color": "black", "width": 1},
        },
        text=[f"μ_{i}" for i in range(len(centroids))],
        textposition="top center",
    )
)

fig_3d.update_layout(
    scene={
        "xaxis_title": "Principal Component 1",
        "yaxis_title": "Principal Component 2",
        "zaxis_title": "Principal Component 3",
    },
    height=600,
    margin={"l": 0, "r": 0, "t": 20, "b": 20},
)

st.plotly_chart(fig_3d, use_container_width=True, key="fig_3d_latent")

st.divider()

# Scree Plot
st.subheader("PCA Variance Explained (Scree Plot)")
fig_scree = go.Figure()
comp_indices = [f"PC {i+1}" for i in range(len(scree_var))]

fig_scree.add_trace(
    go.Bar(
        x=comp_indices,
        y=scree_var * 100,
        name="Individual Explained Variance (%)",
        marker_color="#3b82f6",
    )
)
fig_scree.add_trace(
    go.Scatter(
        x=comp_indices,
        y=cum_var * 100,
        name="Cumulative Explained Variance (%)",
        mode="lines+markers",
        marker_color="#10b981",
        line={"width": 3},
    )
)
fig_scree.add_hline(
    y=90.0,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="90% Variance Target",
)

fig_scree.update_layout(
    xaxis_title="Principal Component",
    yaxis_title="Explained Variance Ratio (%)",
    height=400,
    margin={"l": 20, "r": 20, "t": 30, "b": 20},
)
st.plotly_chart(fig_scree, use_container_width=True, key="fig_scree_plot")
