"""Test that all modules can be imported correctly."""

def test_import_argprism():
    """Test main package import."""
    import argprism
    assert hasattr(argprism, '__version__')

def test_import_classifier():
    """Test classifier module import."""
    from argprism.classifier import ARGClassifier, load_classifier
    assert callable(ARGClassifier)
    assert callable(load_classifier)

def test_import_analysis():
    """Test analysis module import."""
    import argprism.analysis
    # Check if the module has expected components
    assert hasattr(argprism.analysis, 'diamond_mapping')
    assert hasattr(argprism.analysis, 'reporter')

def test_import_cli():
    """Test CLI module import."""
    from argprism.cli import main
    assert callable(main)

def test_import_core():
    """Test core module import."""
    import argprism.core
    assert hasattr(argprism.core, 'pipeline')
    assert hasattr(argprism.core, 'device_manager')

def test_import_io():
    """Test IO module import."""
    import argprism.io
    assert hasattr(argprism.io, 'sequence_io')
    assert hasattr(argprism.io, 'file_paths')

def test_import_utils():
    """Test utils module import."""
    import argprism.utils
    assert hasattr(argprism.utils, 'console')

if __name__ == "__main__":
    test_import_argprism()
    test_import_classifier()
    test_import_analysis()
    test_import_cli()
    test_import_core()
    test_import_io()
    test_import_utils()
    print("All imports successful!")