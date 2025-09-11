"""Integration tests for the complete data pipeline."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from marketregimeml.data.loaders.base import MarketDataLoader
from marketregimeml.features.statistical import StatisticalFeatures
from marketregimeml.features.volatility import VolatilityFeatures
from marketregimeml.features.entropy import EntropyFeatures
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.models import EnsembleRegimeDetector


class TestDataPipelineIntegration:
    """Integration tests for the complete data processing pipeline."""

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        np.random.seed(42)
        dates = pd.date_range(
            "2023-01-01", periods=252, freq="D"
        )  # 1 year of daily data

        # Simulate regime-switching market data
        n = len(dates)
        regimes = []
        prices = [100]

        for i in range(n):
            if i < n // 3:  # Low volatility regime
                regime = 0
                ret = np.random.normal(0.0005, 0.01)
            elif i < 2 * n // 3:  # High volatility regime
                regime = 1
                ret = np.random.normal(0, 0.03)
            else:  # Trending regime
                regime = 2
                ret = np.random.normal(0.002, 0.015)

            regimes.append(regime)
            prices.append(prices[-1] * (1 + ret))

        prices = prices[1:]  # Remove initial price

        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        data["open"] = data["close"] * (1 + np.random.randn(n) * 0.002)
        data["high"] = data[["open", "close"]].max(axis=1) * (
            1 + np.abs(np.random.randn(n) * 0.005)
        )
        data["low"] = data[["open", "close"]].min(axis=1) * (
            1 - np.abs(np.random.randn(n) * 0.005)
        )
        data["volume"] = np.random.randint(1000000, 10000000, n)

        return data, np.array(regimes)

    def test_full_pipeline_with_hmm(self, mock_market_data):
        """Test complete pipeline from data loading to regime detection with HMM."""
        data, true_regimes = mock_market_data

        # Step 1: Feature Engineering
        stat_features = StatisticalFeatures()
        vol_features = VolatilityFeatures()

        # Calculate returns
        returns = data["close"].pct_change().fillna(0)

        # Statistical features
        features = pd.DataFrame(index=data.index)
        features["returns"] = returns
        features["rolling_mean"] = stat_features.rolling_mean(
            returns, window=20
        )
        features["rolling_std"] = stat_features.rolling_std(returns, window=20)
        features["skewness"] = stat_features.rolling_skewness(
            returns, window=20
        )
        features["kurtosis"] = stat_features.rolling_kurtosis(
            returns, window=20
        )

        # Volatility features
        vol_result = vol_features.calculate_features(data, returns=returns)
        features["realized_vol"] = vol_result.get(
            "realized_volatility", returns.rolling(20).std()
        )

        # Clean features
        features = features.fillna(method="ffill").fillna(0)

        # Step 2: Model Training
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(features)

        # Step 3: Prediction
        predictions = model.predict(features)

        # Assertions
        assert len(predictions) == len(data)
        assert predictions.min() >= 0
        assert predictions.max() < 3
        assert len(np.unique(predictions)) <= 3

        # Check that regime transitions occur
        transitions = np.sum(np.diff(predictions) != 0)
        assert transitions > 0  # Should have at least some transitions

    def test_full_pipeline_with_ensemble(self, mock_market_data):
        """Test complete pipeline with ensemble model."""
        data, true_regimes = mock_market_data

        # Feature Engineering
        returns = data["close"].pct_change().fillna(0)

        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": returns.rolling(20).std().fillna(0.01),
                "momentum": returns.rolling(10).mean().fillna(0),
                "volume_ratio": (
                    data["volume"] / data["volume"].rolling(20).mean()
                ).fillna(1),
            }
        )

        # Create ensemble with multiple models
        hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
        gmm = GMMRegimeDetector(n_regimes=3, random_state=42)

        ensemble = EnsembleRegimeDetector(
            models=[hmm, gmm], strategy="voting", n_regimes=3
        )

        # Fit and predict
        ensemble.fit(features)
        predictions = ensemble.predict(features)

        # Assertions
        assert len(predictions) == len(data)
        assert predictions.min() >= 0
        assert predictions.max() < 3

        # Get probabilities
        proba = ensemble.predict_proba(features)
        assert proba.shape == (len(data), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_pipeline_with_data_validation(self, mock_market_data):
        """Test pipeline with data validation steps."""
        data, true_regimes = mock_market_data

        # Introduce some data quality issues
        dirty_data = data.copy()
        dirty_data.iloc[10:15, dirty_data.columns.get_loc("close")] = np.nan
        dirty_data.iloc[50, dirty_data.columns.get_loc("high")] = (
            dirty_data.iloc[50, dirty_data.columns.get_loc("low")] - 1
        )  # Invalid OHLC

        # Data cleaning function
        def clean_ohlcv(df):
            """Clean and validate OHLCV data."""
            # Fill missing values
            df = df.fillna(method="ffill").fillna(method="bfill")

            # Fix OHLC relationships
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

            # Remove zero or negative prices
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].replace(0, np.nan).fillna(method="ffill")
                df[col] = np.where(df[col] <= 0, df[col].shift(1), df[col])

            return df

        # Clean data
        clean_data = clean_ohlcv(dirty_data)

        # Validate cleaned data
        assert not clean_data.isna().any().any()
        assert (clean_data["high"] >= clean_data["low"]).all()
        assert (clean_data["high"] >= clean_data["close"]).all()
        assert (clean_data["low"] <= clean_data["close"]).all()

        # Continue with feature engineering
        returns = clean_data["close"].pct_change().fillna(0)
        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": returns.rolling(20).std().fillna(0.01),
            }
        )

        # Train model
        model = GMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(features)
        predictions = model.predict(features)

        assert len(predictions) == len(clean_data)

    def test_pipeline_with_multiple_timeframes(self, mock_market_data):
        """Test pipeline with multiple timeframe aggregation."""
        data, true_regimes = mock_market_data

        # Create multiple timeframes
        daily_data = data
        weekly_data = data.resample("W").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        monthly_data = data.resample("M").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Features from different timeframes
        daily_returns = daily_data["close"].pct_change().fillna(0)
        weekly_returns = weekly_data["close"].pct_change().fillna(0)
        monthly_returns = monthly_data["close"].pct_change().fillna(0)

        # Align features to daily timeframe
        features = pd.DataFrame(index=daily_data.index)
        features["daily_returns"] = daily_returns
        features["daily_vol"] = daily_returns.rolling(20).std().fillna(0.01)

        # Add weekly features (forward fill to daily)
        features["weekly_returns"] = weekly_returns.reindex(
            daily_data.index, method="ffill"
        ).fillna(0)
        features["weekly_vol"] = (
            weekly_returns.rolling(4)
            .std()
            .reindex(daily_data.index, method="ffill")
            .fillna(0.01)
        )

        # Add monthly features
        features["monthly_returns"] = monthly_returns.reindex(
            daily_data.index, method="ffill"
        ).fillna(0)

        # Train model with multi-timeframe features
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(features)
        predictions = model.predict(features)

        assert len(predictions) == len(daily_data)
        assert len(np.unique(predictions)) <= 3

    def test_pipeline_with_feature_selection(self, mock_market_data):
        """Test pipeline with feature selection step."""
        data, true_regimes = mock_market_data

        # Generate many features
        returns = data["close"].pct_change().fillna(0)

        features = pd.DataFrame(index=data.index)

        # Add various features
        for window in [5, 10, 20, 50]:
            features[f"returns_{window}"] = (
                returns.rolling(window).mean().fillna(0)
            )
            features[f"vol_{window}"] = (
                returns.rolling(window).std().fillna(0.01)
            )
            features[f"skew_{window}"] = (
                returns.rolling(window).skew().fillna(0)
            )
            features[f"kurt_{window}"] = (
                returns.rolling(window).kurt().fillna(3)
            )

        # Add volume features
        features["volume_ratio"] = (
            data["volume"] / data["volume"].rolling(20).mean()
        ).fillna(1)
        features["volume_std"] = (
            data["volume"].rolling(20).std().fillna(data["volume"].std())
        )

        # Feature selection based on variance
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=0.0001)
        selected_features = selector.fit_transform(features)
        selected_feature_names = [
            features.columns[i]
            for i in range(len(features.columns))
            if selector.get_support()[i]
        ]

        selected_df = pd.DataFrame(
            selected_features,
            columns=selected_feature_names,
            index=features.index,
        )

        # Train model with selected features
        model = GMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(selected_df)
        predictions = model.predict(selected_df)

        assert len(predictions) == len(data)
        assert len(selected_df.columns) <= len(features.columns)

    def test_pipeline_with_caching(self, mock_market_data):
        """Test pipeline with feature caching."""
        data, true_regimes = mock_market_data

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "features_cache.pkl"

            # First run - compute features
            returns = data["close"].pct_change().fillna(0)
            features = pd.DataFrame(
                {
                    "returns": returns,
                    "volatility": returns.rolling(20).std().fillna(0.01),
                    "momentum": returns.rolling(10).mean().fillna(0),
                }
            )

            # Save features to cache
            features.to_pickle(cache_path)

            # Second run - load from cache
            cached_features = pd.read_pickle(cache_path)

            # Verify cached features match
            pd.testing.assert_frame_equal(features, cached_features)

            # Train model with cached features
            model = HMMRegimeDetector(n_regimes=3, random_state=42)
            model.fit(cached_features)
            predictions = model.predict(cached_features)

            assert len(predictions) == len(data)

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        # Test with invalid data
        invalid_data = pd.DataFrame(
            {
                "close": [100, 99, -5, 101, 102],  # Negative price
                "volume": [1000, 2000, 3000, 4000, 5000],
            }
        )

        # Should handle negative prices
        returns = invalid_data["close"].pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": returns.rolling(2).std().fillna(0.01),
            }
        )

        # Model should still work with cleaned features
        model = GMMRegimeDetector(n_regimes=2, random_state=42)
        model.fit(features)
        predictions = model.predict(features)

        assert len(predictions) == len(invalid_data)

    def test_pipeline_with_streaming_data(self, mock_market_data):
        """Test pipeline with streaming/incremental data."""
        data, true_regimes = mock_market_data

        # Split data into initial and streaming portions
        initial_size = 100
        initial_data = data.iloc[:initial_size]
        streaming_data = data.iloc[initial_size:]

        # Initial training
        returns = initial_data["close"].pct_change().fillna(0)
        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": returns.rolling(20).std().fillna(0.01),
            }
        )

        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(features)

        # Incremental predictions
        all_predictions = []

        for i in range(len(streaming_data)):
            # Get new data point
            current_data = data.iloc[: initial_size + i + 1]
            current_returns = current_data["close"].pct_change().fillna(0)

            # Update features
            current_features = pd.DataFrame(
                {
                    "returns": current_returns,
                    "volatility": current_returns.rolling(20)
                    .std()
                    .fillna(0.01),
                }
            )

            # Predict only the latest point
            latest_features = current_features.iloc[-1:]
            prediction = model.predict(latest_features)
            all_predictions.append(prediction[0])

        assert len(all_predictions) == len(streaming_data)

    def test_pipeline_performance_metrics(self, mock_market_data):
        """Test pipeline with performance metrics calculation."""
        data, true_regimes = mock_market_data

        # Feature engineering
        returns = data["close"].pct_change().fillna(0)
        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": returns.rolling(20).std().fillna(0.01),
                "momentum": returns.rolling(10).mean().fillna(0),
            }
        )

        # Train-test split
        split_idx = len(features) * 3 // 4
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        _ = true_regimes[:split_idx]  # train_regimes - not used in this test
        _ = true_regimes[split_idx:]  # test_regimes - not used in this test

        # Train model
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(train_features)

        # Predict
        train_pred = model.predict(train_features)
        test_pred = model.predict(test_features)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, confusion_matrix

        # Since regime labels might be permuted, we can't directly compare
        # Instead, check that predictions are valid
        assert len(train_pred) == len(train_features)
        assert len(test_pred) == len(test_features)
        assert train_pred.min() >= 0 and train_pred.max() < 3
        assert test_pred.min() >= 0 and test_pred.max() < 3

        # Check model scores
        train_score = model.score(train_features)
        test_score = model.score(test_features)

        assert isinstance(train_score, float)
        assert isinstance(test_score, float)
