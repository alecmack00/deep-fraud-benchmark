# Enterprise Fraud Detection & Sequence Benchmark

[![CI Tests](https://github.com/alecmack00/deep-fraud-benchmark/actions/workflows/ci.yaml/badge.svg)](https://github.com/alecmack00/deep-fraud-benchmark/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end production benchmark and comparative framework for transaction fraud detection. This platform benchmarks **classical tabular gradient boosted trees and linear models** against **deep sequence representations (Bidirectional LSTM and Transformer Encoder)** on massive chronological transaction streams.

---

## System Architecture

```
                                  IEEE-CIS / PaySim Stream
                         [ 98% Train ]    [ 1% Dev ]    [ 1% Test ]
                                             │
                                             ▼
                   ┌──────────────────────────────────────────────────┐
                   │    Strict Leakage-Free Preprocessing (Train Only) │
                   │    - Correlation Pruning (>90%)                  │
                   │    - Cyclical Hour/Day Sin-Cos Signals           │
                   │    - Median Imputers & Standard Scaler           │
                   └─────────────────────────┬────────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
   ┌───────────────────────────────────┐                   ┌───────────────────────────────────┐
   │ Unsupervised Latent Representations│                  │ Chronological Sequence Windows    │
   │ - IncrementalPCA (90% var)        │                   │ - Grouped by card_id entity       │
   │ - MiniBatchKMeans (Centroid dist) │                   │ - Sliding Window L = 20 events    │
   └─────────────────┬─────────────────┘                   │ - Tensors: [B, L, D] + pad masks  │
                     │                                     └─────────────────┬─────────────────┘
        ┌────────────┼────────────┐                                          │
        ▼            ▼            ▼                             ┌────────────┴────────────┐
   ┌─────────┐  ┌─────────┐  ┌──────────┐                       ▼                         ▼
   │ XGBoost │  │  Random │  │Calibrated│               ┌──────────────┐          ┌──────────────┐
   │  (Hist) │  │  Forest │  │Linear SVM│               │ BiLSTM Model │          │ Transformer  │
   └─────────┘  └─────────┘  └──────────┘               │ (Attn Pool)  │          │ ([CLS] Token)│
                                                        └──────────────┘          └──────────────┘
                                                                │                         │
                                                                └────────────┬────────────┘
                                                                             ▼
                                                               ┌───────────────────────────┐
                                                               │ Binary Focal Loss         │
                                                               │ alpha=0.75, gamma=2.0     │
                                                               └───────────────────────────┘
```

---

## Key Highlights & Guarantees

1. **Zero Lookahead & Entity Leakage**:
   - **Strict Out-of-Time Cutoff**: Chronological ordering on `TransactionDT`.
   - **Big-Data Split Ratios**: 98% Train, 1% Dev, 1% Held-Out Test. Dev and Test sets provide hundreds of positive fraud instances for low-variance statistical estimation while maximizing temporal history in Train.
   - **Sequestered Test Set**: Test split remains strictly held-out until all hyperparameter optimization is complete.
   - **Purged Group TimeSeries Cross-Validation**: 5-fold CV grouped by `card_id` that purges boundary transactions within a 24-hour buffer for entities active in validation folds.
2. **Imbalance-Aware Optimization**:
   - Class imbalance (~3.5% positive fraud) is handled without raw accuracy bias.
   - Models are tuned and early-stopped on **PR-AUC (Average Precision)**, **ROC-AUC**, and **F1-Score**.
3. **Deep Sequence Modeling**:
   - **Sliding History**: $L=20$ consecutive chronological transactions per entity with zero-padding and boolean `padding_mask`.
   - **Bidirectional LSTM**: 2 layers with query-free sequence attention pooling over masked valid steps.
   - **Transformer Encoder**: Injects sinusoidal positional embeddings, prepends a learnable `[CLS]` token, passes through 3 Pre-LN encoder layers, and extracts `[CLS]` sequence representation for classification.
   - **Binary Focal Loss**: $\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ with $\alpha=0.75, \gamma=2.0$.
4. **Bayesian Hyperparameter Optimization**:
   - Powered by Optuna using the Tree-structured Parzen Estimator (TPE) and MedianPruner.
5. **Interactive Comparison Dashboard**:
   - Multi-page Streamlit application connecting directly to MLflow metrics, interactive 3D PCA WebGL latent space scatter, and a financial loss threshold calculator.

---

## Repository Structure (still working on this)


---

## Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/alecmack00/deep-fraud-benchmark.git
cd deep-fraud-sequence-benchmark

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note for macOS users: `brew install libomp` is required for XGBoost OpenMP acceleration).*

### 2. Run the Benchmark Pipeline (Still working on this)

## Testing & Verification (Still working on this)

## Sample Benchmark Results  (Still working on this)


## License
MIT License.
