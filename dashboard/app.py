"""
Enterprise Fraud Detection & Sequence Benchmark - Interactive Comparison Dashboard.
Streamlit Multi-Page Entrypoint.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Enterprise Fraud Benchmark",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium enterprise UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
    }
    .badge-fraud {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-normal {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Enterprise Fraud Detection & Sequence Benchmark")
st.markdown("""
    **End-to-End Comparative Evaluation Framework**: Classical Tabular Baselines (*XGBoost, Random Forest, Calibrated SVM*)
    versus Deep Sequence Modeling (*Bidirectional LSTM, Transformer Encoder with [CLS] Token*).
    """)

# Top Metrics Overview
col1, col2, col3, col4 = st.columns(4)

leaderboard_path = Path("models/artifacts/leaderboard.csv")
if leaderboard_path.exists():
    df_lead = pd.read_csv(leaderboard_path)
    best_pr_auc_row = df_lead.loc[df_lead["PR-AUC"].idxmax()]
    best_roc_auc_row = df_lead.loc[df_lead["ROC-AUC"].idxmax()]
    best_f1_row = df_lead.loc[df_lead["F1 (Fraud)"].idxmax()]
    fastest_row = df_lead.loc[df_lead["Latency (ms/sample)"].idxmin()]

    with col1:
        st.metric(
            "Top PR-AUC Model",
            best_pr_auc_row["Model"],
            f"{best_pr_auc_row['PR-AUC']:.4f}",
        )
    with col2:
        st.metric(
            "Top ROC-AUC Model",
            best_roc_auc_row["Model"],
            f"{best_roc_auc_row['ROC-AUC']:.4f}",
        )
    with col3:
        st.metric(
            "Fastest Inference",
            fastest_row["Model"],
            f"{fastest_row['Latency (ms/sample)']:.2f} ms/tx",
        )
    with col4:
        st.metric(
            "Highest Fraud F1",
            best_f1_row["Model"],
            f"{best_f1_row['F1 (Fraud)']:.4f}",
        )
else:
    with col1:
        st.metric("Primary Dataset", "IEEE-CIS / PaySim", "590K Transactions")
    with col2:
        st.metric("Temporal Split", "98% / 1% / 1%", "No-Leakage Purged")
    with col3:
        st.metric("Sequence Window L", "20 Events", "BiLSTM & Transformer")
    with col4:
        st.metric("Primary Metric", "PR-AUC", "Imbalance-Aware")

st.divider()

st.subheader("Architectural Benchmark Overview")

tab1, tab2, tab3 = st.tabs(
    ["System Architecture", "Zero-Leakage Guarantee", "Deep Sequence Formulation"]
)

with tab1:
    st.markdown("""
        ```
        +-----------------------------------------------------------------------------------+
        |                                IEEE-CIS / PaySim Stream                           |
        |              [ 98% Train Split ]          [ 1% Dev Split ]  [ 1% Held-Out Test ]  |
        +-----------------------------------------------------------------------------------+
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │   Leakage-Free Preprocessing (Fit ONLY on Train) │
                        │   - High Correlation Pruning (>90%)              │
                        │   - Cyclical Hour/Day Sin-Cos Encodings          │
                        │   - Median Imputation & Standard Scaling         │
                        └─────────────────────────┬────────────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  ▼                                                               ▼
        ┌───────────────────────────────────┐               ┌───────────────────────────────────┐
        │ Tabular & Unsupervised Features   │               │ Chronological Sequence Tensors    │
        │ - IncrementalPCA (90% var)        │               │ - Grouped by card_id entity       │
        │ - MiniBatchKMeans (Centroid dist) │               │ - Sliding Window L = 20           │
        └─────────────────┬─────────────────┘               │ - Tensor: [B, L, D]               │
                          │                                 └─────────────────┬─────────────────┘
             ┌────────────┼────────────┐                                      │
             ▼            ▼            ▼                         ┌────────────┴────────────┐
        ┌─────────┐  ┌─────────┐  ┌──────────┐                   ▼                         ▼
        │ XGBoost │  │  Random │  │Calibrated│           ┌──────────────┐          ┌──────────────┐
        │  (Hist) │  │  Forest │  │Linear SVM│           │ BiLSTM Model │          │ Transformer  │
        └─────────┘  └─────────┘  └──────────┘           │ (Attn Pool)  │          │ ([CLS] Token)│
                                                         └──────────────┘          └──────────────┘
        ```
        """)

with tab2:
    st.markdown("""
        - **Strict Out-of-Time Cutoff**: Transactions strictly sorted by `TransactionDT`. Lookahead leakage is mathematically prevented.
        - **Sequestered Test Split**: The 1% held-out test split is locked and untouched during feature engineering, PCA, clustering, and Optuna tuning.
        - **Purged Group Cross-Validation**: Boundary transactions within a 24-hour buffer for entities active in validation folds are purged.
        """)

with tab3:
    st.markdown("""
        - **Sliding History**: $L=20$ consecutive chronological transactions per cardholder.
        - **BiLSTM with Attention Pooling**: Aggregates variable-length transaction events using query-free temporal attention while masking zero-padded history steps.
        - **Transformer Encoder**: Injects sinusoidal positional embeddings, prepends a learnable `[CLS]` classification token, and optimizes a **Binary Focal Loss** ($\alpha=0.75, \\gamma=2.0$) to counteract the ~3.5% positive fraud rarity.
        """)

st.sidebar.success("Navigate through the sidebar pages to explore the benchmark.")
