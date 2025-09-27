"""Integration tests for all models to improve coverage."""

import pytest
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from marketregimeml.models import (
    HMMRegimeDetector,
    GMMRegimeDetector,
    GARCHRegimeDetector,
    MSGARCHRegimeDetector,
    RandomForestRegimeClassifier,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
    EnsembleRegimeDetector,
)
from marketregimeml.evaluation import ModelEvaluator
from marketregimeml.data.loaders.oanda import OANDADataLoader
from marketregimeml.features import (
    VolatilityFeatures,
    StatisticalFeatures,
    TechnicalIndicators,
)
from marketregimeml.config_loader import ConfigLoader


class TestAllModelsIntegration:
    """Integration tests for all models."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame(
            {
                "returns": np.random.randn(n) * 0.02,
                "volatility": np.abs(np.random.randn(n)) * 0.01 + 0.01,
                "volume": np.abs(np.random.randn(n)) * 0.1 + 1.0,
                "momentum": np.random.randn(n) * 0.03,
            }
        )
        return data

    @pytest.fixture
    def ohlcv_data(self):
        """Generate OHLCV data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        data = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.001),
                "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
                "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
                "close": close,
                "volume": 1000000 * (1 + np.random.randn(n) * 0.3),
            },
            index=dates,
        )
        return data

    def test_hmm_model(self, sample_data):
        """Test HMM model."""
        model = HMMRegimeDetector(n_regimes=3, n_iter=50)
        model.fit(sample_data)

        # Test predictions
        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(0 <= p < 3 for p in predictions)

        # Test probabilities
        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # Test fuzzy predictions
        model_fuzzy = HMMRegimeDetector(n_regimes=3, fuzzy_matching=True)
        model_fuzzy.fit(sample_data)
        fuzzy_result = model_fuzzy.predict_fuzzy(sample_data)
        assert "crisp" in fuzzy_result
        assert "confidence" in fuzzy_result
        assert len(fuzzy_result["crisp"]) == len(sample_data)
        assert all(0 <= c <= 1 for c in fuzzy_result["confidence"])

        # Test regime transitions
        regimes = model.predict(sample_data)
        transitions = model.analyze_regime_transitions(regimes)
        assert "transition_matrix" in transitions
        assert transitions["transition_matrix"].shape == (3, 3)

    def test_gmm_model(self, sample_data):
        """Test GMM model."""
        model = GMMRegimeDetector(n_regimes=3, max_iter=50)
        model.fit(sample_data)

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 3)

        # Test different covariance types
        for cov_type in ["spherical", "diag", "full"]:
            model = GMMRegimeDetector(n_regimes=2, covariance_type=cov_type)
            model.fit(sample_data)
            preds = model.predict(sample_data)
            assert len(preds) == len(sample_data)

    def test_garch_model(self, sample_data):
        """Test GARCH model."""
        model = GARCHRegimeDetector(n_regimes=3, p=1, q=1)
        model.fit(sample_data)

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 3)

    def test_msgarch_model(self, sample_data):
        """Test MS-GARCH model."""
        model = MSGARCHRegimeDetector(n_regimes=2)
        model.fit(sample_data)

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 2)

    def test_random_forest_model(self, sample_data):
        """Test Random Forest model."""
        model = RandomForestRegimeClassifier(n_regimes=3, n_estimators=10)
        model.fit(sample_data)

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 3)

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == sample_data.shape[1]

    def test_xgboost_model(self, sample_data):
        """Test XGBoost model."""
        model = XGBoostRegimeClassifier(n_regimes=3, n_estimators=10)
        model.fit(sample_data)

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 3)

    def test_svm_model(self, sample_data):
        """Test SVM model."""
        model = SVMRegimeClassifier(n_regimes=3, probability=True)
        model.fit(sample_data)

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 3)

    # Deep learning models removed - not implemented

    def test_ensemble_model(self, sample_data):
        """Test Ensemble model."""
        # Test with default models
        model = EnsembleRegimeDetector(n_regimes=3, voting="hard")
        model.fit(sample_data)

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        proba = model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 3)

        # Test different voting methods
        for voting_type in ["hard", "soft"]:
            model = EnsembleRegimeDetector(n_regimes=3, voting=voting_type)
            model.fit(sample_data)
            preds = model.predict(sample_data)
            assert len(preds) == len(sample_data)

        # Test with custom models
        # Create models with names as required by VotingEnsemble
        named_models = [
            ("hmm", HMMRegimeDetector(n_regimes=3)),
            ("gmm", GMMRegimeDetector(n_regimes=3)),
        ]
        model = EnsembleRegimeDetector(models=named_models, voting="soft", n_regimes=3)
        model.fit(sample_data)
        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

    def test_model_evaluator(self, sample_data):
        """Test ModelEvaluator."""
        # Fit a model
        model = HMMRegimeDetector(n_regimes=3)
        model.fit(sample_data)

        # Create evaluator with model
        evaluator = ModelEvaluator(model)

        # Evaluate
        metrics = evaluator.evaluate_model(sample_data)

        # Check metrics exist - they are nested under categories
        assert "clustering_metrics" in metrics
        assert "model_diagnostics" in metrics
        assert "regime_analysis" in metrics

        # Check clustering metrics
        clustering = metrics["clustering_metrics"]
        assert "silhouette_score" in clustering
        assert "davies_bouldin_index" in clustering
        assert "calinski_harabasz_index" in clustering

        # Check model diagnostics
        diagnostics = metrics["model_diagnostics"]
        assert "aic" in diagnostics
        assert "bic" in diagnostics

        # Check regime analysis
        regime = metrics["regime_analysis"]
        assert "stability" in regime

        # Check that we have a regime quality index
        assert "regime_quality_index" in metrics

    def test_volatility_features(self, ohlcv_data):
        """Test volatility features."""
        vol_features = VolatilityFeatures()

        # Test different volatility measures
        yz = vol_features.yang_zhang(
            ohlcv_data["open"],
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
        )
        assert len(yz) == len(ohlcv_data)

        gk = vol_features.garman_klass(
            ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"]
        )
        assert len(gk) == len(ohlcv_data)

        park = vol_features.parkinson(ohlcv_data["high"], ohlcv_data["low"])
        assert len(park) == len(ohlcv_data)

    def test_statistical_features(self, sample_data):
        """Test statistical features."""
        stats_features = StatisticalFeatures()

        # Test different statistical measures
        returns = pd.Series(sample_data["returns"].values)

        skew = stats_features.rolling_skewness(returns, window=20)
        assert len(skew) == len(returns)

        kurt = stats_features.rolling_kurtosis(returns, window=20)
        assert len(kurt) == len(returns)

        autocorr = stats_features.autocorrelation(returns, lag=1)
        assert len(autocorr) == len(returns)

    def test_technical_indicators(self, ohlcv_data):
        """Test technical indicators."""
        tech_indicators = TechnicalIndicators()

        # Test RSI
        rsi = tech_indicators.rsi(ohlcv_data["close"])
        assert len(rsi) == len(ohlcv_data)
        assert all(0 <= r <= 100 for r in rsi.dropna())

        # Test MACD
        macd_line, signal, histogram = tech_indicators.macd(ohlcv_data["close"])
        assert len(macd_line) == len(ohlcv_data)
        assert len(signal) == len(ohlcv_data)
        assert len(histogram) == len(ohlcv_data)

        # Test Bollinger Bands
        upper, middle, lower = tech_indicators.bollinger_bands(ohlcv_data["close"])
        assert len(upper) == len(ohlcv_data)
        assert len(middle) == len(ohlcv_data)
        assert len(lower) == len(ohlcv_data)

    def test_config_loader(self):
        """Test ConfigLoader."""
        config = ConfigLoader()

        # Test loading configs
        try:
            markets = config.get_market_config()
            assert isinstance(markets, dict)
        except FileNotFoundError:
            # Config files may not exist in test environment
            pass

        # Test environment variable processing
        import os

        os.environ["TEST_VAR"] = "test_value"
        # ConfigLoader processes env vars through config strings
        # Test that env vars are resolved during config loading
        test_config = {"test_key": "${TEST_VAR}"}
        resolved = config._process_env_vars(test_config)
        assert resolved["test_key"] == "test_value"

    def test_arbitrary_regimes(self, sample_data):
        """Test arbitrary regime counts."""
        # Test different regime counts
        for n_regimes in [2, 3, 5, 7, 9]:
            model = HMMRegimeDetector(n_regimes=n_regimes)
            model.fit(sample_data)

            predictions = model.predict(sample_data)
            assert all(0 <= p < n_regimes for p in predictions)

            proba = model.predict_proba(sample_data)
            assert proba.shape == (len(sample_data), n_regimes)

    def test_regime_optimization(self, sample_data):
        """Test automatic regime optimization."""
        model = HMMRegimeDetector(
            auto_optimize_regimes=True, min_regimes=2, max_regimes=5
        )
        model.fit(sample_data)

        # Check that regime count was optimized
        assert 2 <= model.n_regimes <= 5

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)
