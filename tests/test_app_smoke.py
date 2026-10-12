"""
test_app_smoke.py - Smoke test for Streamlit app compilation and module imports
"""

import importlib
import pytest


def test_app_imports_and_components():
    """Verify all application components can be imported without syntax or import errors."""
    import app
    assert hasattr(app, "get_base_data")
    assert hasattr(app, "get_cached_model")

    # Verify model and data can be loaded via app's cached functions
    df = app.get_base_data()
    assert not df.empty

    model, meta, metrics = app.get_cached_model()
    assert model is not None
    assert "best_model_name" in meta
    assert not metrics.empty
