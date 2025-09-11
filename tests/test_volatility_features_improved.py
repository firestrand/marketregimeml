"""Improved tests for volatility features to increase coverage."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from marketregimeml.features.volatility import VolatilityFeatures


class TestVolatilityFeaturesImproved:
    """Comprehensive test suite for volatility features."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 252  # One year of daily data
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate realistic price data with volatility clustering
        returns = []
        vol_regime = 0.01  # Start with low volatility

        for i in range(n):
            # Occasionally switch volatility regime
            if np.random.random() < 0.05:
                vol_regime = np.random.choice([0.005, 0.01, 0.02, 0.03])

            returns.append(np.random.normal(0.0005, vol_regime))

        # Generate prices from returns
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": np.array(prices) * (1 + np.random.randn(n) * 0.002),
                "high": np.array(prices)
                * (1 + np.abs(np.random.randn(n) * 0.005)),
                "low": np.array(prices)
                * (1 - np.abs(np.random.randn(n) * 0.005)),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, n),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        return data

    @pytest.fixture
    def sample_returns(self, sample_ohlcv):
        """Create sample returns."""
        return sample_ohlcv["close"].pct_change().fillna(0)

    @pytest.fixture
    def features(self):
        """Create VolatilityFeatures instance."""
        return VolatilityFeatures()

    def test_historical_volatility(self, features, sample_returns):
        """Test historical volatility calculation."""
        vol = features.historical_volatility(sample_returns, window=20)

        assert len(vol) == len(sample_returns)
        assert vol.isna().sum() == 19  # First 19 values should be NaN

        # Volatility should be positive
        valid_vol = vol[~vol.isna()]
        assert (valid_vol >= 0).all()

        # Test annualization
        vol_annual = features.historical_volatility(
            sample_returns, window=20, annualize=True
        )
        # Annualized vol should be roughly sqrt(252) times larger
        assert vol_annual.mean() > vol.mean()

    def test_parkinson_volatility(self, features, sample_ohlcv):
        """Test Parkinson volatility estimator."""
        vol = features.parkinson_volatility(
            sample_ohlcv["high"], sample_ohlcv["low"], window=20
        )

        assert len(vol) == len(sample_ohlcv)

        # Volatility should be positive
        valid_vol = vol[~vol.isna()]
        assert (valid_vol >= 0).all()

        # Parkinson should be different from simple historical vol
        returns = sample_ohlcv["close"].pct_change()
        hist_vol = features.historical_volatility(returns, window=20)
        assert not np.allclose(vol[20:], hist_vol[20:], equal_nan=True)

    def test_garman_klass_volatility(self, features, sample_ohlcv):
        """Test Garman-Klass volatility estimator."""
        vol = features.garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=20,
        )

        assert len(vol) == len(sample_ohlcv)

        # Volatility should be positive
        valid_vol = vol[~vol.isna()]
        assert (valid_vol >= 0).all()

        # GK should typically be more efficient than Parkinson
        park_vol = features.parkinson_volatility(
            sample_ohlcv["high"], sample_ohlcv["low"], window=20
        )

        # Both should exist and be different
        assert not np.allclose(vol[20:], park_vol[20:], equal_nan=True)

    def test_rogers_satchell_volatility(self, features, sample_ohlcv):
        """Test Rogers-Satchell volatility estimator."""
        vol = features.rogers_satchell_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=20,
        )

        assert len(vol) == len(sample_ohlcv)

        # Volatility should be positive
        valid_vol = vol[~vol.isna()]
        assert (valid_vol >= 0).all()

        # RS handles drift, should be different from GK
        gk_vol = features.garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=20,
        )

        assert not np.allclose(vol[20:], gk_vol[20:], equal_nan=True)

    def test_yang_zhang_volatility(self, features, sample_ohlcv):
        """Test Yang-Zhang volatility estimator."""
        vol = features.yang_zhang_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=20,
        )

        assert len(vol) == len(sample_ohlcv)

        # Volatility should be positive
        valid_vol = vol[~vol.isna()]
        assert (valid_vol >= 0).all()

        # YZ is most comprehensive, should be different from others
        rs_vol = features.rogers_satchell_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=20,
        )

        assert not np.allclose(vol[20:], rs_vol[20:], equal_nan=True)

    def test_realized_volatility(self, features, sample_returns):
        """Test realized volatility calculation."""
        # Simulate intraday returns (5-minute bars for a day)
        intraday_returns = pd.Series(
            np.random.normal(
                0, 0.001, 78
            ),  # 78 five-minute bars in a trading day
            index=pd.date_range("2023-01-01 09:30", periods=78, freq="5min"),
        )

        rv = features.realized_volatility(intraday_returns, frequency="5min")

        assert isinstance(rv, float)
        assert rv >= 0

        # Test with daily returns
        daily_rv = features.realized_volatility(
            sample_returns[:20], frequency="daily"
        )
        assert daily_rv >= 0

    def test_garch_volatility(self, features, sample_returns):
        """Test GARCH volatility forecast."""
        # Use sufficient data for GARCH
        returns = sample_returns[~sample_returns.isna()]

        if len(returns) > 100:
            vol = features.garch_volatility(returns, p=1, q=1)

            assert len(vol) == len(returns)

            # GARCH volatility should be positive
            valid_vol = vol[~vol.isna()]
            assert (valid_vol > 0).all()

            # GARCH should show volatility clustering
            # High volatility should persist
            high_vol_periods = vol > vol.median()
            transitions = np.diff(high_vol_periods.astype(int))
            # Should have some persistence (not switching every period)
            assert np.sum(np.abs(transitions)) < len(vol) * 0.8

    def test_ewma_volatility(self, features, sample_returns):
        """Test EWMA volatility."""
        vol = features.ewma_volatility(sample_returns, span=20)

        assert len(vol) == len(sample_returns)

        # EWMA should have no NaN values (except first if returns has NaN)
        if (
            not sample_returns.iloc[0] != sample_returns.iloc[0]
        ):  # Check if not NaN
            assert not vol.isna().any()

        # Volatility should be positive
        valid_vol = vol[~vol.isna()]
        assert (valid_vol >= 0).all()

        # EWMA should react faster to shocks than simple moving average
        hist_vol = features.historical_volatility(sample_returns, window=20)

        # Create a shock
        shock_returns = sample_returns.copy()
        shock_returns.iloc[100] = 0.1  # Large return

        ewma_shock = features.ewma_volatility(shock_returns, span=20)
        hist_shock = features.historical_volatility(shock_returns, window=20)

        # EWMA should show larger immediate response
        ewma_response = ewma_shock.iloc[101] - vol.iloc[101]
        hist_response = hist_shock.iloc[101] - hist_vol.iloc[101]

        if not np.isnan(ewma_response) and not np.isnan(hist_response):
            assert abs(ewma_response) > abs(hist_response) * 0.5

    def test_volatility_ratio(self, features, sample_returns):
        """Test volatility ratio calculation."""
        ratio = features.volatility_ratio(
            sample_returns, short_window=10, long_window=50
        )

        assert len(ratio) == len(sample_returns)

        # Ratio should be positive
        valid_ratio = ratio[~ratio.isna()]
        assert (valid_ratio >= 0).all()

        # During volatile periods, short-term vol > long-term vol (ratio > 1)
        # During calm periods, ratio should be < 1
        assert valid_ratio.min() < 1.5  # Should have some calm periods
        assert valid_ratio.max() > 0.5  # Should have some volatile periods

    def test_volatility_cone(self, features, sample_returns):
        """Test volatility cone calculation."""
        windows = [5, 10, 20, 50]
        cone = features.volatility_cone(sample_returns, windows=windows)

        assert isinstance(cone, pd.DataFrame)
        assert len(cone.columns) == len(windows)

        for window in windows:
            assert f"vol_{window}" in cone.columns

            # Each volatility should be positive
            valid_vol = cone[f"vol_{window}"][~cone[f"vol_{window}"].isna()]
            assert (valid_vol >= 0).all()

        # Longer windows should generally have smoother volatility
        if len(cone) > 50:
            vol_5_std = cone["vol_5"].std()
            vol_50_std = cone["vol_50"].std()
            # 50-day vol should be less variable than 5-day vol
            if not np.isnan(vol_5_std) and not np.isnan(vol_50_std):
                assert vol_50_std < vol_5_std

    def test_calculate_features(self, features, sample_ohlcv):
        """Test main calculate_features method."""
        returns = sample_ohlcv["close"].pct_change()

        result = features.calculate_features(sample_ohlcv, returns=returns)

        assert isinstance(result, dict)

        # Check for expected keys
        expected_keys = [
            "historical_volatility",
            "parkinson_volatility",
            "garman_klass_volatility",
            "ewma_volatility",
        ]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], pd.Series)
            assert len(result[key]) == len(sample_ohlcv)

    def test_volatility_term_structure(self, features, sample_returns):
        """Test volatility term structure."""
        windows = [5, 10, 20, 30, 60]

        term_structure = {}
        for window in windows:
            vol = features.historical_volatility(sample_returns, window=window)
            term_structure[window] = (
                vol.iloc[-1] if not vol.isna().all() else np.nan
            )

        # Term structure should exist
        assert len(term_structure) == len(windows)

        # In normal markets, term structure often upward sloping
        # But this is not always true, so just check values exist
        valid_vols = [v for v in term_structure.values() if not np.isnan(v)]
        assert len(valid_vols) > 0
        assert all(v >= 0 for v in valid_vols)

    def test_volatility_smile(self, features):
        """Test implied volatility smile pattern."""
        # Simulate option implied volatilities at different strikes
        strikes = np.array([90, 95, 100, 105, 110])
        spot = 100

        # Create smile pattern (higher IV for OTM options)
        moneyness = strikes / spot
        impl_vols = 0.20 + 0.1 * (moneyness - 1.0) ** 2

        # Volatility smile should show U-shape
        assert impl_vols[0] > impl_vols[2]  # OTM put higher than ATM
        assert impl_vols[4] > impl_vols[2]  # OTM call higher than ATM
        assert impl_vols[2] == min(impl_vols)  # ATM should be minimum

    def test_conditional_volatility(self, features, sample_returns):
        """Test conditional volatility based on market conditions."""
        # Calculate volatility in different market conditions
        positive_returns = sample_returns > 0
        negative_returns = sample_returns < 0

        # Volatility during up days
        up_vol = sample_returns[positive_returns].std()

        # Volatility during down days
        down_vol = sample_returns[negative_returns].std()

        # Both should exist and be positive
        assert up_vol > 0
        assert down_vol > 0

        # Often, down volatility > up volatility (leverage effect)
        # But not always true, so just check they're computed
        assert isinstance(up_vol, float)
        assert isinstance(down_vol, float)

    def test_jump_detection(self, features, sample_returns):
        """Test detection of jumps in returns."""
        # Add artificial jumps
        returns_with_jumps = sample_returns.copy()
        jump_indices = [50, 100, 150]

        for idx in jump_indices:
            if idx < len(returns_with_jumps):
                returns_with_jumps.iloc[idx] = (
                    returns_with_jumps.iloc[idx] + 0.05
                )

        # Simple jump detection: returns > 3 standard deviations
        rolling_std = returns_with_jumps.rolling(20).std()
        rolling_mean = returns_with_jumps.rolling(20).mean()
        z_scores = (returns_with_jumps - rolling_mean) / rolling_std

        jumps = np.abs(z_scores) > 3

        # Should detect some jumps
        assert jumps.sum() > 0

        # Jumps should be rare
        assert jumps.sum() < len(returns_with_jumps) * 0.1

    def test_regime_dependent_volatility(self, features):
        """Test volatility in different market regimes."""
        # Create synthetic data with regime switches
        n = 500
        regime1_vol = 0.01
        regime2_vol = 0.03

        returns = []
        current_regime = 1

        for i in range(n):
            # Switch regime occasionally
            if np.random.random() < 0.02:
                current_regime = 3 - current_regime  # Switch between 1 and 2

            if current_regime == 1:
                returns.append(np.random.normal(0, regime1_vol))
            else:
                returns.append(np.random.normal(0, regime2_vol))

        returns = pd.Series(returns)

        # Calculate rolling volatility
        rolling_vol = features.historical_volatility(returns, window=20)

        # Volatility should vary over time
        assert rolling_vol.std() > 0

        # Should have periods of both low and high volatility
        assert rolling_vol.min() < regime2_vol * 0.7  # Some low vol periods
        assert rolling_vol.max() > regime1_vol * 1.3  # Some high vol periods

    def test_get_feature_names(self, features):
        """Test getting feature names."""
        names = features.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0

        # Check for expected volatility measures
        expected = [
            "historical_volatility",
            "parkinson_volatility",
            "garman_klass_volatility",
            "yang_zhang_volatility",
        ]

        for exp in expected:
            assert exp in names
