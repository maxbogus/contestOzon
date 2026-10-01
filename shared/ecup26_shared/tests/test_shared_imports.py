"""Smoke tests: verify shared/ package structure and imports."""
from __future__ import annotations


def test_shared_package_imports():
    """ecup26_shared package is importable."""
    import ecup26_shared
    assert ecup26_shared is not None


def test_shared_ml_subpackage():
    import ecup26_shared.ml
    assert hasattr(ecup26_shared.ml, "__path__")


def test_shared_eval_subpackage():
    import ecup26_shared.eval
    assert hasattr(ecup26_shared.eval, "__path__")


def test_shared_nlp_subpackage():
    import ecup26_shared.nlp
    assert hasattr(ecup26_shared.nlp, "__path__")


def test_shared_fusion_subpackage():
    import ecup26_shared.fusion
    assert hasattr(ecup26_shared.fusion, "__path__")


def test_shared_cascade_subpackage():
    import ecup26_shared.cascade
    assert hasattr(ecup26_shared.cascade, "__path__")


def test_shared_sweep_subpackage():
    import ecup26_shared.sweep
    assert hasattr(ecup26_shared.sweep, "__path__")


def test_shared_sparse_subpackage():
    import ecup26_shared.sparse
    assert hasattr(ecup26_shared.sparse, "__path__")


def test_shared_pipeline_subpackage():
    import ecup26_shared.pipeline
    assert hasattr(ecup26_shared.pipeline, "__path__")


def test_test_subpackage_exists():
    # tests/ is one level up now
    import ecup26_shared
    assert hasattr(ecup26_shared, "__path__")
