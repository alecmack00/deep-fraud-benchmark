"""
Page 2: Curve & Calibration Viewer with Financial Loss Threshold Slider.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

st.title("Curve & Calibration Viewer with Financial Loss Analysis")
st.markdown(
    "Analyze Precision-Recall, ROC, and Reliability curves, and dynamically optimize the decision threshold "
    "to minimize real-world fraud and friction costs."
)


@st.cache_data
def load_model_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    candidate_paths = [
        Path(f"models/artifacts/{model_name}/test_predictions.npz"),
        Path(
            f"models/artifacts/{model_name.replace('_Baseline', '').replace('_Sequence', '')}/test_predictions.npz"
        ),
        Path(f"models/artifacts/{model_name}_Baseline/test_predictions.npz"),
        Path(f"models/artifacts/{model_name}_Sequence/test_predictions.npz"),
    ]
    for p in candidate_paths:
        if p.exists():
            data = np.load(p)
            return data["y_true"], data["y_pred_proba"]

    # Fallback simulation
    np.random.seed(42)
    n_samples = 3000
    y_true = (np.random.uniform(0, 1, size=n_samples) < 0.035).astype(int)
    fraud_scores = np.random.beta(5, 2, size=n_samples)
    normal_scores = np.random.beta(1, 15, size=n_samples)
    y_pred_proba = np.where(y_true == 1, fraud_scores, normal_scores)
    return y_true, y_pred_proba


# Model Selector
available_models = [
    "XGBoost",
    "RandomForest",
    "CalibratedSVM",
    "BiLSTM",
    "Transformer",
    "Transformer_Sequence",
    "BiLSTM_Sequence",
    "XGBoost_Baseline",
    "RandomForest_Baseline",
    "CalibratedSVM_Baseline",
]
# Filter to existing or standard list
present_dirs = (
    [d.name for d in Path("models/artifacts").iterdir() if d.is_dir()]
    if Path("models/artifacts").exists()
    else []
)
model_options = [m for m in available_models if m in present_dirs] or [
    "XGBoost",
    "RandomForest",
    "CalibratedSVM",
    "BiLSTM",
    "Transformer",
]

selected_model = st.sidebar.selectbox("Select Model for Analysis:", model_options)
y_true, y_pred_proba = load_model_predictions(selected_model)

# Sidebar Financial Parameters
st.sidebar.header("Financial Cost Model")
fn_cost = st.sidebar.number_input(
    "Cost of Missed Fraud (False Negative)",
    min_value=10.0,
    max_value=5000.0,
    value=500.0,
    step=50.0,
)
fp_cost = st.sidebar.number_input(
    "Cost of Customer Friction (False Positive)",
    min_value=1.0,
    max_value=500.0,
    value=25.0,
    step=5.0,
)

threshold = st.sidebar.slider(
    "Decision Threshold (tau)", min_value=0.01, max_value=0.99, value=0.35, step=0.01
)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Precision-Recall Curve ({selected_model})")
    p, r, _ = precision_recall_curve(y_true, y_pred_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=r,
            y=p,
            mode="lines",
            name="PR Curve",
            line={"color": "#2563eb", "width": 3},
        )
    )
    base_rate = float(np.mean(y_true))
    fig_pr.add_hline(
        y=base_rate,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Base Rate ({base_rate:.3f})",
    )
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=380,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    st.plotly_chart(fig_pr, use_container_width=True, key=f"pr_chart_{selected_model}")

with col2:
    st.subheader(f"Receiver Operating Characteristic ({selected_model})")
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve",
            line={"color": "#10b981", "width": 3},
        )
    )
    fig_roc.add_shape(
        type="line", line={"dash": "dash", "color": "gray"}, x0=0, x1=1, y0=0, y1=1
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    st.plotly_chart(
        fig_roc, use_container_width=True, key=f"roc_chart_{selected_model}"
    )

st.divider()

st.subheader("Interactive Financial Loss Optimization")

y_pred_bin = (y_pred_proba >= threshold).astype(int)
cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

total_fn_cost = fn * fn_cost
total_fp_cost = fp * fp_cost
total_financial_loss = total_fn_cost + total_fp_cost
loss_per_tx = total_financial_loss / max(1, len(y_true))

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Selected Threshold", f"{threshold:.2f}")
col_m2.metric(
    "Missed Fraud Cost ($FN)", f"${total_fn_cost:,.0f}", f"{fn} missed frauds"
)
col_m3.metric(
    "Customer Friction Cost ($FP)", f"${total_fp_cost:,.0f}", f"{fp} false flags"
)
col_m4.metric(
    "Net Financial Loss", f"${total_financial_loss:,.0f}", f"${loss_per_tx:.2f} / tx"
)


@st.cache_data
def compute_loss_curve(
    y_t: np.ndarray, y_p: np.ndarray, fn_c: float, fp_c: float
) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    threshold_range = np.linspace(0.02, 0.98, 97)
    loss_curve = []
    fn_costs = []
    fp_costs = []
    for th in threshold_range:
        pred_th = (y_p >= th).astype(int)
        c_m = confusion_matrix(y_t, pred_th, labels=[0, 1])
        _, _fp, _fn, _ = c_m.ravel()
        cost_fn = _fn * fn_c
        cost_fp = _fp * fp_c
        loss_curve.append(cost_fn + cost_fp)
        fn_costs.append(cost_fn)
        fp_costs.append(cost_fp)
    return threshold_range, loss_curve, fn_costs, fp_costs


threshold_range, loss_curve, fn_costs, fp_costs = compute_loss_curve(
    y_true, y_pred_proba, fn_cost, fp_cost
)

optimal_idx = int(np.argmin(loss_curve))
optimal_th = float(threshold_range[optimal_idx])
min_loss = float(loss_curve[optimal_idx])

fig_loss = go.Figure()
fig_loss.add_trace(
    go.Scatter(
        x=threshold_range,
        y=loss_curve,
        mode="lines",
        name="Total Loss ($)",
        line={"color": "#dc2626", "width": 3},
    )
)
fig_loss.add_trace(
    go.Scatter(
        x=threshold_range,
        y=fn_costs,
        mode="lines",
        name="Missed Fraud Cost ($FN)",
        line={"color": "#f97316", "dash": "dot"},
    )
)
fig_loss.add_trace(
    go.Scatter(
        x=threshold_range,
        y=fp_costs,
        mode="lines",
        name="Friction Cost ($FP)",
        line={"color": "#3b82f6", "dash": "dot"},
    )
)
fig_loss.add_vline(
    x=threshold,
    line_dash="solid",
    line_color="#10b981",
    annotation_text=f"Current ({threshold:.2f})",
)
fig_loss.add_vline(
    x=optimal_th,
    line_dash="dash",
    line_color="#7c3aed",
    annotation_text=f"Optimal ({optimal_th:.2f}: ${min_loss:,.0f})",
)

fig_loss.update_layout(
    title="Expected Financial Loss vs Decision Threshold",
    xaxis_title="Classification Threshold",
    yaxis_title="Total Cost ($)",
    height=420,
    hovermode="x unified",
)
st.plotly_chart(fig_loss, use_container_width=True, key=f"loss_chart_{selected_model}")
