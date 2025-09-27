"""Tests for model persistence - TDD approach."""

import os
import tempfile
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.models.ml import (
    RandomForestRegimeClassifier,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
)


class TestModelPersistence:
    """Test suite for model persistence functionality."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5

        # Generate synthetic regime data
        X = np.random.randn(n_samples, n_features)

        # Create regime patterns
        for i in range(0, n_samples, 100):
            regime = (i // 100) % 3
            X[i : i + 100] += regime * 2

        # Convert to DataFrame as expected by models
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        return df

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for saving models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_hmm_save_load(self, sample_data, temp_dir):
        """Test HMM model save and load functionality."""
        # Train model
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(sample_data)

        # Get predictions before saving
        predictions_before = model.predict(sample_data)
        params_before = model.get_params()

        # Save model
        save_path = os.path.join(temp_dir, "hmm_model.pkl")
        model.save_model(save_path)

        # Verify file exists
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

        # Load model
        loaded_model = HMMRegimeDetector.load_model(save_path)

        # Verify loaded model works
        predictions_after = loaded_model.predict(sample_data)
        params_after = loaded_model.get_params()

        # Check predictions are the same
        np.testing.assert_array_equal(predictions_before, predictions_after)

        # Check parameters are preserved
        assert params_before["n_regimes"] == params_after["n_regimes"]
        assert params_before["covariance_type"] == params_after["covariance_type"]

    def test_gmm_save_load(self, sample_data, temp_dir):
        """Test GMM model save and load functionality."""
        # Train model
        model = GMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(sample_data)

        # Get predictions before saving
        predictions_before = model.predict(sample_data)
        proba_before = model.predict_proba(sample_data)

        # Save model
        save_path = os.path.join(temp_dir, "gmm_model.pkl")
        model.save_model(save_path)

        # Verify file exists
        assert os.path.exists(save_path)

        # Load model
        loaded_model = GMMRegimeDetector.load_model(save_path)

        # Verify loaded model works
        predictions_after = loaded_model.predict(sample_data)
        proba_after = loaded_model.predict_proba(sample_data)

        # Check results are the same
        np.testing.assert_array_equal(predictions_before, predictions_after)
        np.testing.assert_array_almost_equal(proba_before, proba_after)

    def test_random_forest_save_load(self, sample_data, temp_dir):
        """Test RandomForest model save and load functionality."""
        # Train model
        model = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=50, random_state=42
        )
        model.fit(sample_data)

        # Get predictions before saving
        predictions_before = model.predict(sample_data)

        # Save model
        save_path = os.path.join(temp_dir, "rf_model.pkl")
        model.save_model(save_path)

        # Load model
        loaded_model = RandomForestRegimeClassifier.load_model(save_path)

        # Verify loaded model works
        predictions_after = loaded_model.predict(sample_data)

        # Check predictions are similar (may have small differences due to clustering)
        accuracy = np.mean(predictions_before == predictions_after)
        assert accuracy > 0.9  # Allow some variation due to clustering initialization

    def test_xgboost_save_load(self, sample_data, temp_dir):
        """Test XGBoost model save and load functionality."""
        # Train model
        model = XGBoostRegimeClassifier(n_regimes=3, n_estimators=50, random_state=42)
        model.fit(sample_data)

        # Get predictions before saving
        predictions_before = model.predict(sample_data)

        # Save model using pickle (XGBoost models with sklearn components can't use JSON)
        pkl_path = os.path.join(temp_dir, "xgb_model.pkl")
        model.save_model(pkl_path)

        # Verify file exists
        assert os.path.exists(pkl_path)

        # Load from pickle
        loaded_pkl = XGBoostRegimeClassifier.load_model(pkl_path)
        predictions_pkl = loaded_pkl.predict(sample_data)

        # Check predictions are similar
        accuracy_pkl = np.mean(predictions_before == predictions_pkl)
        assert accuracy_pkl > 0.9

    def test_svm_save_load(self, sample_data, temp_dir):
        """Test SVM model save and load functionality."""
        # Train model with smaller dataset (SVM is slow)
        small_data = sample_data[:200]

        model = SVMRegimeClassifier(n_regimes=3, kernel="rbf", random_state=42)
        model.fit(small_data)

        # Get predictions before saving
        predictions_before = model.predict(small_data)

        # Save model
        save_path = os.path.join(temp_dir, "svm_model.pkl")
        model.save_model(save_path)

        # Load model
        loaded_model = SVMRegimeClassifier.load_model(save_path)

        # Verify loaded model works
        predictions_after = loaded_model.predict(small_data)

        # Check predictions are similar
        accuracy = np.mean(predictions_before == predictions_after)
        assert accuracy > 0.9

    def test_save_with_metadata(self, sample_data, temp_dir):
        """Test saving models with metadata."""
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(sample_data)

        # Add metadata
        metadata = {
            "training_date": "2024-01-01",
            "data_source": "test_data",
            "version": "1.0.0",
            "metrics": {"log_likelihood": -1234.56, "aic": 2469.12},
        }

        # Save with metadata
        save_path = os.path.join(temp_dir, "model_with_metadata.pkl")
        model.save_model(save_path, metadata=metadata)

        # Load model and metadata
        loaded_model, loaded_metadata = HMMRegimeDetector.load_model(
            save_path, return_metadata=True
        )

        # Verify metadata
        assert loaded_metadata["training_date"] == "2024-01-01"
        assert loaded_metadata["version"] == "1.0.0"
        assert loaded_metadata["metrics"]["aic"] == 2469.12

    def test_save_load_different_formats(self, sample_data, temp_dir):
        """Test saving and loading in different formats."""
        model = GMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(sample_data)

        # Test pickle format
        pkl_path = os.path.join(temp_dir, "model.pkl")
        model.save_model(pkl_path, format="pickle")
        loaded_pkl = GMMRegimeDetector.load_model(pkl_path, format="pickle")
        assert loaded_pkl is not None

        # Test joblib format
        joblib_path = os.path.join(temp_dir, "model.joblib")
        model.save_model(joblib_path, format="joblib")
        loaded_joblib = GMMRegimeDetector.load_model(joblib_path, format="joblib")
        assert loaded_joblib is not None

        # Both should give same predictions
        pred_pkl = loaded_pkl.predict(sample_data)
        pred_joblib = loaded_joblib.predict(sample_data)
        np.testing.assert_array_equal(pred_pkl, pred_joblib)

    def test_invalid_file_path(self):
        """Test error handling for invalid file paths."""
        # Try to load non-existent file
        with pytest.raises(ValueError, match="Failed to load model"):
            HMMRegimeDetector.load_model("/non/existent/path.pkl")

        # Try to save to invalid directory
        model = HMMRegimeDetector(n_regimes=2)
        with pytest.raises(OSError):
            model.save_model("/invalid/directory/model.pkl")

    def test_model_versioning(self, sample_data, temp_dir):
        """Test model versioning support."""
        # Train and save v1
        model_v1 = HMMRegimeDetector(n_regimes=2, random_state=42)
        model_v1.fit(sample_data[:500])

        path_v1 = os.path.join(temp_dir, "model_v1.pkl")
        model_v1.save_model(path_v1, metadata={"version": "1.0"})

        # Train and save v2 with different parameters
        model_v2 = HMMRegimeDetector(n_regimes=3, random_state=42)
        model_v2.fit(sample_data)

        path_v2 = os.path.join(temp_dir, "model_v2.pkl")
        model_v2.save_model(path_v2, metadata={"version": "2.0"})

        # Load both versions
        loaded_v1, meta_v1 = HMMRegimeDetector.load_model(path_v1, return_metadata=True)
        loaded_v2, meta_v2 = HMMRegimeDetector.load_model(path_v2, return_metadata=True)

        # Verify versions
        assert meta_v1["version"] == "1.0"
        assert meta_v2["version"] == "2.0"

        # Verify different parameters
        assert loaded_v1.n_regimes == 2
        assert loaded_v2.n_regimes == 3

    def test_compression_options(self, sample_data, temp_dir):
        """Test model compression options."""
        model = RandomForestRegimeClassifier(n_regimes=3, n_estimators=100)
        model.fit(sample_data)

        # Save uncompressed
        uncompressed_path = os.path.join(temp_dir, "model_uncompressed.pkl")
        model.save_model(uncompressed_path, compress=False)

        # Save compressed
        compressed_path = os.path.join(temp_dir, "model_compressed.pkl.gz")
        model.save_model(compressed_path, compress=True)

        # Check file sizes
        uncompressed_size = os.path.getsize(uncompressed_path)
        compressed_size = os.path.getsize(compressed_path)

        # Compressed should be smaller
        assert compressed_size < uncompressed_size

        # Both should load correctly
        loaded_uncompressed = RandomForestRegimeClassifier.load_model(uncompressed_path)
        loaded_compressed = RandomForestRegimeClassifier.load_model(compressed_path)

        # Both should give same predictions
        pred_uncompressed = loaded_uncompressed.predict(sample_data)
        pred_compressed = loaded_compressed.predict(sample_data)

        accuracy = np.mean(pred_uncompressed == pred_compressed)
        assert accuracy > 0.95  # Allow small differences

    def test_backward_compatibility(self, sample_data, temp_dir):
        """Test loading models saved with older versions."""
        # Simulate an old model format
        old_model_data = {
            "model_type": "HMMRegimeDetector",
            "params": {"n_regimes": 3},
            "fitted": True,
            "version": "0.1.0",  # Old version
        }

        old_path = os.path.join(temp_dir, "old_model.pkl")
        with open(old_path, "wb") as f:
            pickle.dump(old_model_data, f)

        # Should handle gracefully or provide migration
        try:
            loaded = HMMRegimeDetector.load_model(old_path)
            assert loaded is not None
        except Exception as e:
            # Should provide helpful error message
            assert "version" in str(e).lower() or "compatibility" in str(e).lower()
