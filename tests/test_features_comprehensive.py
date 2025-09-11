"""Comprehensive tests for feature engineering modules following SOLID, DRY, KISS."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from marketregimeml.features.technical import TechnicalIndicators
from marketregimeml.features.volatility import VolatilityFeatures
from marketregimeml.features.statistical import StatisticalFeatures
from marketregimeml.features.entropy import EntropyFeatures
from marketregimeml.features.engine import FeatureEngine


class TestTechnicalIndicatorsComprehensive:
    """Comprehensive tests for technical indicators - Single Responsibility."""

    @pytest.fixture
    def ohlcv_data(self):
        """Generate OHLCV data - DRY principle."""
        np.random.seed(42)
        n = 200  # Enough for all indicators
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate realistic price movement
        returns = np.random.randn(n) * 0.02
        close = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.002),
                "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
                "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
                "close": close,
                "volume": 1000000 * (1 + np.random.randn(n) * 0.1),
            },
            index=dates,
        )

    def test_moving_averages(self, ohlcv_data):
        """Test moving average calculations - KISS principle."""
        tech = TechnicalIndicators()

        # Simple Moving Average
        sma = tech.sma(ohlcv_data["close"], period=20)
        assert len(sma) == len(ohlcv_data)
        assert sma.iloc[19:].notna().all()  # Should have values after period

        # Exponential Moving Average
        ema = tech.ema(ohlcv_data["close"], period=20)
        assert len(ema) == len(ohlcv_data)
        assert ema.iloc[0:].notna().any()  # EMA starts earlier than SMA

        # Weighted Moving Average
        wma = tech.wma(ohlcv_data["close"], period=20)
        assert len(wma) == len(ohlcv_data)

    def test_momentum_indicators(self, ohlcv_data):
        """Test momentum indicators."""
        tech = TechnicalIndicators()

        # RSI
        rsi = tech.rsi(ohlcv_data["close"], period=14)
        assert (rsi[~rsi.isna()] >= 0).all()
        assert (rsi[~rsi.isna()] <= 100).all()

        # Stochastic
        k, d = tech.stochastic(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            period=14,
        )
        assert (k[~k.isna()] >= 0).all()
        assert (k[~k.isna()] <= 100).all()

        # Williams %R
        williams = tech.williams_r(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            period=14,
        )
        assert (williams[~williams.isna()] >= -100).all()
        assert (williams[~williams.isna()] <= 0).all()

    def test_trend_indicators(self, ohlcv_data):
        """Test trend indicators."""
        tech = TechnicalIndicators()

        # MACD
        macd, signal, histogram = tech.macd(ohlcv_data["close"])
        assert len(macd) == len(ohlcv_data)
        assert len(signal) == len(ohlcv_data)
        assert len(histogram) == len(ohlcv_data)

        # ADX
        adx = tech.adx(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            period=14,
        )
        assert (adx[~adx.isna()] >= 0).all()
        assert (adx[~adx.isna()] <= 100).all()

    def test_volatility_indicators(self, ohlcv_data):
        """Test volatility indicators."""
        tech = TechnicalIndicators()

        # Bollinger Bands
        upper, middle, lower = tech.bollinger_bands(ohlcv_data["close"])
        valid = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

        # ATR
        atr = tech.atr(
            ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"]
        )
        assert (atr[~atr.isna()] >= 0).all()

        # Keltner Channels
        k_upper, k_middle, k_lower = tech.keltner_channels(
            ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"]
        )
        valid_k = ~(k_upper.isna() | k_middle.isna() | k_lower.isna())
        assert (k_upper[valid_k] >= k_middle[valid_k]).all()

    def test_volume_indicators(self, ohlcv_data):
        """Test volume indicators."""
        tech = TechnicalIndicators()

        # OBV
        obv = tech.obv(ohlcv_data["close"], ohlcv_data["volume"])
        assert len(obv) == len(ohlcv_data)
        assert obv.iloc[0] == ohlcv_data["volume"].iloc[0]

        # Volume SMA
        vol_sma = tech.volume_sma(ohlcv_data["volume"], period=20)
        assert len(vol_sma) == len(ohlcv_data)

        # VWAP
        vwap = tech.vwap(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
        )
        assert len(vwap) == len(ohlcv_data)


class TestVolatilityFeaturesComprehensive:
    """Comprehensive tests for volatility features - Single Responsibility."""

    @pytest.fixture
    def price_data(self):
        """Generate price data - DRY principle."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate with volatility clusters
        vol = np.ones(n) * 0.01
        vol[50:70] = 0.03  # High vol period
        vol[120:140] = 0.005  # Low vol period

        returns = np.random.randn(n) * vol
        close = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.001),
                "high": close * (1 + np.abs(np.random.randn(n)) * 0.003),
                "low": close * (1 - np.abs(np.random.randn(n)) * 0.003),
                "close": close,
            },
            index=dates,
        )

    def test_historical_volatility(self, price_data):
        """Test historical volatility calculations - KISS principle."""
        vol = VolatilityFeatures()

        # Simple volatility
        simple_vol = vol.historical_volatility(price_data["close"], window=20)
        assert len(simple_vol) == len(price_data)
        assert (simple_vol[~simple_vol.isna()] >= 0).all()

        # Annualized volatility
        annual_vol = vol.historical_volatility(
            price_data["close"], window=20, annualize=True
        )
        assert (annual_vol[~annual_vol.isna()] >= 0).all()
        # Annualized should be larger
        valid = ~(simple_vol.isna() | annual_vol.isna())
        assert (annual_vol[valid] >= simple_vol[valid]).all()

    def test_parkinson_volatility(self, price_data):
        """Test Parkinson volatility estimator."""
        vol = VolatilityFeatures()

        park_vol = vol.parkinson(
            price_data["high"], price_data["low"], window=20
        )

        assert len(park_vol) == len(price_data)
        assert (park_vol[~park_vol.isna()] >= 0).all()

    def test_garman_klass_volatility(self, price_data):
        """Test Garman-Klass volatility estimator."""
        vol = VolatilityFeatures()

        gk_vol = vol.garman_klass(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            window=20,
        )

        assert len(gk_vol) == len(price_data)
        assert (gk_vol[~gk_vol.isna()] >= 0).all()

    def test_rogers_satchell_volatility(self, price_data):
        """Test Rogers-Satchell volatility estimator."""
        vol = VolatilityFeatures()

        rs_vol = vol.rogers_satchell(
            price_data["open"],
            price_data["high"],
            price_data["low"],
            price_data["close"],
            window=20,
        )

        assert len(rs_vol) == len(price_data)
        assert (rs_vol[~rs_vol.isna()] >= 0).all()

    def test_yang_zhang_volatility(self, price_data):
        """Test Yang-Zhang volatility estimator."""
        vol = VolatilityFeatures()

        yz_vol = vol.yang_zhang(
            price_data["open"],
            price_data["high"],
            price_data["low"],
            price_data["close"],
            window=20,
        )

        assert len(yz_vol) == len(price_data)
        assert (yz_vol[~yz_vol.isna()] >= 0).all()

    def test_volatility_comparison(self, price_data):
        """Test that different volatility estimators are consistent."""
        vol = VolatilityFeatures()
        window = 30

        # Calculate all volatility measures
        hist_vol = vol.historical_volatility(price_data["close"], window)
        park_vol = vol.parkinson(price_data["high"], price_data["low"], window)
        gk_vol = vol.garman_klass(
            price_data["high"], price_data["low"], price_data["close"], window
        )

        # They should be correlated
        valid = ~(hist_vol.isna() | park_vol.isna() | gk_vol.isna())
        if valid.sum() > 10:
            corr_hp = np.corrcoef(hist_vol[valid], park_vol[valid])[0, 1]
            corr_hg = np.corrcoef(hist_vol[valid], gk_vol[valid])[0, 1]

            assert corr_hp > 0.5  # Should be positively correlated
            assert corr_hg > 0.5


class TestStatisticalFeaturesComprehensive:
    """Comprehensive tests for statistical features - Single Responsibility."""

    @pytest.fixture
    def returns_data(self):
        """Generate returns data - DRY principle."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate with different statistical properties
        returns = np.random.randn(n) * 0.01
        # Add skewness
        returns[50:70] = np.random.exponential(0.01, 20) - 0.01
        # Add fat tails
        returns[120:130] = np.random.standard_t(3, 10) * 0.02

        return pd.Series(returns, index=dates)

    def test_rolling_moments(self, returns_data):
        """Test rolling statistical moments - KISS principle."""
        stat = StatisticalFeatures()

        # Mean
        rolling_mean = stat.rolling_mean(returns_data, window=20)
        assert len(rolling_mean) == len(returns_data)

        # Variance
        rolling_var = stat.rolling_variance(returns_data, window=20)
        assert (rolling_var[~rolling_var.isna()] >= 0).all()

        # Standard deviation
        rolling_std = stat.rolling_std(returns_data, window=20)
        assert (rolling_std[~rolling_std.isna()] >= 0).all()

    def test_higher_moments(self, returns_data):
        """Test skewness and kurtosis."""
        stat = StatisticalFeatures()

        # Skewness
        skew = stat.rolling_skewness(returns_data, window=30)
        assert len(skew) == len(returns_data)
        # Skewness can be positive or negative
        assert skew[~skew.isna()].min() < 1
        assert skew[~skew.isna()].max() > -1

        # Kurtosis
        kurt = stat.rolling_kurtosis(returns_data, window=30)
        assert len(kurt) == len(returns_data)
        # Excess kurtosis for normal is 0
        assert kurt[~kurt.isna()].min() > -3

    def test_autocorrelation_features(self, returns_data):
        """Test autocorrelation-based features."""
        stat = StatisticalFeatures()

        # Autocorrelation
        acf = stat.autocorrelation(returns_data, window=50, lag=1)
        assert len(acf) == len(returns_data)
        assert (acf[~acf.isna()] >= -1).all()
        assert (acf[~acf.isna()] <= 1).all()

        # Partial autocorrelation
        pacf = stat.partial_autocorrelation(returns_data, window=50, lag=1)
        assert len(pacf) == len(returns_data)

    def test_distribution_tests(self, returns_data):
        """Test distribution testing features."""
        stat = StatisticalFeatures()

        # Jarque-Bera test
        jb_stat = stat.jarque_bera_test(returns_data, window=50)
        assert len(jb_stat) == len(returns_data)
        assert (jb_stat[~jb_stat.isna()] >= 0).all()

        # Shapiro-Wilk test
        sw_stat = stat.shapiro_wilk_test(returns_data, window=50)
        assert len(sw_stat) == len(returns_data)

    def test_tail_measures(self, returns_data):
        """Test tail risk measures."""
        stat = StatisticalFeatures()

        # Tail ratio
        tail_ratio = stat.tail_ratio(returns_data, window=50)
        assert len(tail_ratio) == len(returns_data)
        assert (tail_ratio[~tail_ratio.isna()] > 0).all()

        # Extreme value statistics
        max_val = stat.rolling_max(returns_data, window=20)
        min_val = stat.rolling_min(returns_data, window=20)

        assert (max_val >= min_val).all()


class TestEntropyFeaturesComprehensive:
    """Comprehensive tests for entropy features - Single Responsibility."""

    @pytest.fixture
    def time_series_data(self):
        """Generate time series data - DRY principle."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Mix of deterministic and random components
        t = np.linspace(0, 10, n)
        signal = np.sin(t) * 0.5 + np.random.randn(n) * 0.1

        return pd.Series(signal, index=dates)

    def test_shannon_entropy(self, time_series_data):
        """Test Shannon entropy calculation - KISS principle."""
        entropy = EntropyFeatures()

        shannon = entropy.shannon_entropy(time_series_data, window=50, bins=10)

        assert len(shannon) == len(time_series_data)
        assert (shannon[~shannon.isna()] >= 0).all()

        # Test with different bin sizes
        shannon_5 = entropy.shannon_entropy(
            time_series_data, window=50, bins=5
        )
        shannon_20 = entropy.shannon_entropy(
            time_series_data, window=50, bins=20
        )

        # More bins generally means higher entropy
        valid = ~(shannon_5.isna() | shannon_20.isna())
        if valid.sum() > 10:
            assert shannon_20[valid].mean() >= shannon_5[valid].mean()

    def test_sample_entropy(self, time_series_data):
        """Test sample entropy calculation."""
        entropy = EntropyFeatures()

        sample_ent = entropy.sample_entropy(
            time_series_data, window=50, m=2, r=0.2 * time_series_data.std()
        )

        assert len(sample_ent) == len(time_series_data)
        valid = sample_ent[~sample_ent.isna()]
        if len(valid) > 0:
            assert (valid >= 0).all()

    def test_approximate_entropy(self, time_series_data):
        """Test approximate entropy calculation."""
        entropy = EntropyFeatures()

        approx_ent = entropy.approximate_entropy(
            time_series_data, window=50, m=2, r=0.2 * time_series_data.std()
        )

        assert len(approx_ent) == len(time_series_data)
        valid = approx_ent[~approx_ent.isna()]
        if len(valid) > 0:
            assert (valid >= 0).all()

    def test_permutation_entropy(self, time_series_data):
        """Test permutation entropy calculation."""
        entropy = EntropyFeatures()

        perm_ent = entropy.permutation_entropy(
            time_series_data, window=50, order=3
        )

        assert len(perm_ent) == len(time_series_data)
        valid = perm_ent[~perm_ent.isna()]
        if len(valid) > 0:
            assert (valid >= 0).all()
            assert (valid <= np.log(6)).all()  # Max for order 3

    def test_multiscale_entropy(self, time_series_data):
        """Test multiscale entropy calculation."""
        entropy = EntropyFeatures()

        mse = entropy.multiscale_entropy(
            time_series_data, window=100, scales=[1, 2, 4]
        )

        assert len(mse) == len(time_series_data)
        assert mse.shape[1] == 3  # One column per scale


class TestFeatureEngineComprehensive:
    """Comprehensive tests for FeatureEngine - Single Responsibility."""

    @pytest.fixture
    def config(self):
        """Feature configuration - DRY principle."""
        return {
            "features": {
                "technical": {
                    "rsi": {"period": 14},
                    "sma": {"period": 20},
                    "ema": {"period": 20},
                },
                "volatility": {
                    "historical": {"window": 20},
                    "parkinson": {"window": 20},
                },
                "statistical": {
                    "skewness": {"window": 30},
                    "kurtosis": {"window": 30},
                },
                "entropy": {"shannon": {"window": 50, "bins": 10}},
            }
        }

    @pytest.fixture
    def market_data(self):
        """Generate market data - DRY principle."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.001),
                "high": close * (1 + np.abs(np.random.randn(n)) * 0.003),
                "low": close * (1 - np.abs(np.random.randn(n)) * 0.003),
                "close": close,
                "volume": 1000000 * (1 + np.random.randn(n) * 0.1),
            },
            index=dates,
        )

    def test_init(self, config):
        """Test initialization - KISS principle."""
        engine = FeatureEngine(config)

        assert engine.config == config
        assert hasattr(engine, "technical")
        assert hasattr(engine, "volatility")
        assert hasattr(engine, "statistical")
        assert hasattr(engine, "entropy")

    def test_calculate_features(self, config, market_data):
        """Test feature calculation."""
        engine = FeatureEngine(config)

        features = engine.calculate_features(market_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) <= len(market_data)  # Some rows lost to NaN

        # Check expected features exist
        expected_features = [
            "rsi",
            "sma",
            "ema",
            "historical",
            "parkinson",
            "skewness",
            "kurtosis",
            "shannon",
        ]
        for feat in expected_features:
            assert any(feat in col for col in features.columns)

    def test_get_feature_info(self, config):
        """Test feature information retrieval."""
        engine = FeatureEngine(config)

        info = engine.get_feature_info()

        assert isinstance(info, list)
        assert len(info) > 0

        for feature_info in info:
            assert "name" in feature_info
            assert "type" in feature_info
            assert "params" in feature_info

    def test_select_features(self, config, market_data):
        """Test feature selection."""
        engine = FeatureEngine(config)
        features = engine.calculate_features(market_data)

        # Variance threshold selection
        selected = engine.select_features(
            features, method="variance", threshold=0.001
        )
        assert selected.shape[1] <= features.shape[1]

        # Correlation threshold selection
        selected = engine.select_features(
            features, method="correlation", threshold=0.9
        )
        assert selected.shape[1] <= features.shape[1]

    def test_scale_features(self, config, market_data):
        """Test feature scaling."""
        engine = FeatureEngine(config)
        features = engine.calculate_features(market_data)

        # Standard scaling
        scaled = engine.scale_features(features, method="standard")
        assert scaled.shape == features.shape
        # Check mean ~0 and std ~1
        assert np.abs(scaled.mean()).max() < 0.1
        assert np.abs(scaled.std() - 1).max() < 0.1

        # MinMax scaling
        scaled = engine.scale_features(features, method="minmax")
        assert (scaled.min() >= 0).all()
        assert (scaled.max() <= 1).all()

    def test_add_custom_feature(self, config, market_data):
        """Test adding custom features."""
        engine = FeatureEngine(config)

        # Define custom feature
        def momentum_feature(data):
            return data["close"].pct_change(10)

        engine.add_custom_feature("momentum_10", momentum_feature)

        features = engine.calculate_features(market_data)
        assert "momentum_10" in features.columns

    def test_feature_importance(self, config, market_data):
        """Test feature importance calculation."""
        engine = FeatureEngine(config)
        features = engine.calculate_features(market_data)

        # Create target (e.g., future returns)
        target = market_data["close"].pct_change().shift(-1)
        target = target.loc[features.index]

        importance = engine.calculate_feature_importance(
            features, target, method="random_forest"
        )

        assert len(importance) == features.shape[1]
        assert importance.sum() > 0
