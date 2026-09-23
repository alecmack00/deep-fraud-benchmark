import importlib
from pathlib import Path


def test_no_page_config_in_subpages():
    """
    Validates that st.set_page_config is only called in app.py and never in subpages,
    preventing StreamlitAPIException crashes in multi-page mode.
    """
    pages_dir = Path("dashboard/pages")
    assert pages_dir.exists()

    for page_file in pages_dir.glob("*.py"):
        content = page_file.read_text()
        assert "st.set_page_config" not in content, (
            f"Violation: {page_file.name} contains st.set_page_config. "
            f"Multi-page apps must only call set_page_config in app.py."
        )

    # Verify app.py contains st.set_page_config
    app_content = Path("dashboard/app.py").read_text()
    assert "st.set_page_config" in app_content


def test_leaderboard_page_loading():
    p1 = importlib.import_module("dashboard.pages.01_leaderboard")
    df = p1.load_leaderboard_data()
    assert not df.empty
    assert "Model" in df.columns
    assert "PR-AUC" in df.columns
    assert "ROC-AUC" in df.columns


def test_curves_page_and_loss_curve():
    p2 = importlib.import_module("dashboard.pages.02_curves_and_calibration")
    y_true, y_pred_proba = p2.load_model_predictions("XGBoost")
    assert len(y_true) == len(y_pred_proba)
    assert len(y_true) > 0

    # Test financial loss computation
    th_range, loss_curve, _fn_costs, _fp_costs = p2.compute_loss_curve(
        y_true, y_pred_proba, fn_c=500.0, fp_c=25.0
    )
    assert len(th_range) == len(loss_curve) == 97
    assert min(loss_curve) >= 0.0


def test_latent_space_dimensions():
    p3 = importlib.import_module("dashboard.pages.03_latent_space")
    coords, _labels, _clusters, centroids, scree_var, cum_var = p3.load_latent_data()
    assert coords.shape[1] == 3
    assert centroids.shape[1] == 3
    assert len(scree_var) == len(cum_var)


def test_latency_profiler_page():
    p4 = importlib.import_module("dashboard.pages.04_latency_profiler")
    df = p4.load_latency_data()
    assert not df.empty
    assert "Latency (ms/sample)" in df.columns
    assert "Model" in df.columns
