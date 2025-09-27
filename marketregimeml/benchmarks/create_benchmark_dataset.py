"""Create benchmark datasets for regime detection testing.

This module creates standardized benchmark datasets using OANDA data
for testing and comparing regime detection algorithms.
"""

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from marketregimeml.data.loaders.oanda import OANDADataLoader
from marketregimeml.features import FeatureEngine
from marketregimeml.utils.logging import get_logger


logger = get_logger(__name__)


class BenchmarkDataset:
    """Creates and manages benchmark datasets for regime detection."""

    def __init__(self, cache_dir: str = "data/benchmarks"):
        """Initialize benchmark dataset creator.

        Args:
            cache_dir: Directory to store benchmark datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.oanda_loader = OANDADataLoader()
        self.feature_engine = FeatureEngine()

    def create_forex_benchmark(
        self,
        pairs: List[str] = None,
        timeframe: str = "H1",
        lookback_days: int = 365,
        save_name: str = "forex_benchmark_1year",
    ) -> Dict[str, pd.DataFrame]:
        """Create FOREX benchmark dataset.

        Args:
            pairs: List of currency pairs (default: major pairs)
            timeframe: OANDA timeframe
            lookback_days: Number of days of historical data
            save_name: Name for saved dataset

        Returns:
            Dictionary of DataFrames with OHLCV and features
        """
        if pairs is None:
            pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]

        benchmark_data = {}

        for pair in pairs:
            logger.info(f"Fetching data for {pair}")

            try:
                # Calculate start date
                from datetime import datetime, timedelta

                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                # Fetch OHLCV data
                ohlcv = self.oanda_loader.fetch_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Calculate features
                features = self.feature_engine.compute_features(ohlcv)

                # Add regime labels (for benchmarking)
                regimes = self._generate_synthetic_regimes(ohlcv)
                features["true_regime"] = regimes

                benchmark_data[pair] = {
                    "ohlcv": ohlcv,
                    "features": features,
                    "metadata": {
                        "pair": pair,
                        "timeframe": timeframe,
                        "start": ohlcv.index[0],
                        "end": ohlcv.index[-1],
                        "samples": len(ohlcv),
                    },
                }

                logger.info(f"Successfully processed {pair}: {len(ohlcv)} samples")

            except Exception as e:
                logger.error(f"Failed to process {pair}: {e}")

        # Save benchmark dataset
        save_path = self.cache_dir / f"{save_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(benchmark_data, f)

        logger.info(f"Saved benchmark dataset to {save_path}")

        return benchmark_data

    def create_multi_timeframe_benchmark(
        self,
        symbol: str = "EUR_USD",
        timeframes: List[str] = None,
        lookback_days: int = 90,
        save_name: str = "multi_timeframe_benchmark",
    ) -> Dict[str, pd.DataFrame]:
        """Create multi-timeframe benchmark dataset.

        Args:
            symbol: Currency pair
            timeframes: List of timeframes
            lookback_days: Number of days of historical data
            save_name: Name for saved dataset

        Returns:
            Dictionary of DataFrames by timeframe
        """
        if timeframes is None:
            timeframes = ["M5", "M15", "H1", "H4", "D"]

        benchmark_data = {}

        for tf in timeframes:
            logger.info(f"Fetching {symbol} at {tf} timeframe")

            try:
                # Note: Using date range instead of limit for better precision

                # Calculate start date
                from datetime import datetime, timedelta

                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                # Fetch data
                ohlcv = self.oanda_loader.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Calculate features
                features = self.feature_engine.compute_features(ohlcv)

                benchmark_data[tf] = {
                    "ohlcv": ohlcv,
                    "features": features,
                    "metadata": {
                        "symbol": symbol,
                        "timeframe": tf,
                        "start": ohlcv.index[0],
                        "end": ohlcv.index[-1],
                        "samples": len(ohlcv),
                    },
                }

                logger.info(f"Processed {tf}: {len(ohlcv)} samples")

            except Exception as e:
                logger.error(f"Failed to process {tf}: {e}")

        # Save benchmark dataset
        save_path = self.cache_dir / f"{save_name}_{symbol}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(benchmark_data, f)

        logger.info(f"Saved multi-timeframe benchmark to {save_path}")

        return benchmark_data

    def create_crisis_periods_benchmark(
        self, save_name: str = "crisis_periods_benchmark"
    ) -> Dict[str, pd.DataFrame]:
        """Create benchmark dataset with known crisis periods.

        Returns:
            Dictionary of crisis period data
        """
        crisis_periods = {
            "covid_2020": {
                "start": "2020-02-01",
                "end": "2020-04-30",
                "pairs": ["EUR_USD", "GBP_USD", "USD_JPY"],
            },
            "brexit_2016": {
                "start": "2016-06-01",
                "end": "2016-07-31",
                "pairs": ["GBP_USD", "EUR_GBP"],
            },
            "normal_2019": {
                "start": "2019-03-01",
                "end": "2019-05-31",
                "pairs": ["EUR_USD", "USD_JPY"],
            },
        }

        benchmark_data = {}

        for period_name, period_info in crisis_periods.items():
            logger.info(f"Processing {period_name}")

            period_data = {}
            for pair in period_info["pairs"]:
                try:
                    # Note: OANDA historical data access may be limited
                    # This is a demonstration of the structure
                    logger.warning(
                        f"Historical data for {period_name} may require specific OANDA subscription"
                    )

                    # For demonstration, fetch recent data
                    ohlcv = self.oanda_loader.fetch_ohlcv(
                        symbol=pair,
                        timeframe="H1",
                        limit=720,  # 30 days of hourly data
                    )

                    features = self.feature_engine.calculate_features(ohlcv)

                    period_data[pair] = {
                        "ohlcv": ohlcv,
                        "features": features,
                        "metadata": {
                            "period": period_name,
                            "pair": pair,
                            "expected_regime": (
                                "volatile"
                                if "covid" in period_name or "brexit" in period_name
                                else "normal"
                            ),
                        },
                    }

                except Exception as e:
                    logger.error(f"Failed to process {pair} for {period_name}: {e}")

            benchmark_data[period_name] = period_data

        # Save benchmark dataset
        save_path = self.cache_dir / f"{save_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(benchmark_data, f)

        logger.info(f"Saved crisis periods benchmark to {save_path}")

        return benchmark_data

    def _generate_synthetic_regimes(
        self,
        ohlcv: pd.DataFrame,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.001,
    ) -> np.ndarray:
        """Generate synthetic regime labels based on simple rules.

        Args:
            ohlcv: OHLCV data
            volatility_threshold: Threshold for high volatility
            trend_threshold: Threshold for trending market

        Returns:
            Array of regime labels (0: ranging, 1: trending, 2: volatile)
        """
        # Calculate returns and volatility
        returns = ohlcv["close"].pct_change()
        volatility = returns.rolling(20).std()
        sma_20 = ohlcv["close"].rolling(20).mean()
        sma_50 = ohlcv["close"].rolling(50).mean()

        # Initialize regimes
        regimes = np.zeros(len(ohlcv))

        # Classify regimes
        for i in range(50, len(ohlcv)):
            vol = volatility.iloc[i]
            trend = (sma_20.iloc[i] - sma_50.iloc[i]) / sma_50.iloc[i]

            if vol > volatility_threshold:
                regimes[i] = 2  # Volatile
            elif abs(trend) > trend_threshold:
                regimes[i] = 1  # Trending
            else:
                regimes[i] = 0  # Ranging

        return regimes

    def load_benchmark(self, name: str) -> Dict:
        """Load a saved benchmark dataset.

        Args:
            name: Name of the benchmark dataset

        Returns:
            Loaded benchmark data
        """
        load_path = self.cache_dir / f"{name}.pkl"

        if not load_path.exists():
            raise FileNotFoundError(f"Benchmark dataset {name} not found")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        logger.info(f"Loaded benchmark dataset from {load_path}")

        return data

    def list_benchmarks(self) -> List[str]:
        """List available benchmark datasets.

        Returns:
            List of benchmark dataset names
        """
        benchmarks = [f.stem for f in self.cache_dir.glob("*.pkl")]
        return benchmarks


def create_standard_benchmarks():
    """Create standard benchmark datasets for testing."""

    benchmark = BenchmarkDataset()

    # Create FOREX benchmark
    logger.info("Creating FOREX benchmark dataset...")
    forex_data = benchmark.create_forex_benchmark(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY"],
        timeframe="H1",
        lookback_days=30,  # Start with 30 days for testing
        save_name="forex_benchmark_30d",
    )

    # Create multi-timeframe benchmark
    logger.info("Creating multi-timeframe benchmark dataset...")
    mtf_data = benchmark.create_multi_timeframe_benchmark(
        symbol="EUR_USD",
        timeframes=["M15", "H1", "H4"],
        lookback_days=30,
        save_name="mtf_benchmark_30d",
    )

    # Summary statistics
    logger.info("\n=== Benchmark Dataset Summary ===")

    for pair, data in forex_data.items():
        meta = data["metadata"]
        logger.info(
            f"{pair}: {meta['samples']} samples from {meta['start']} to {meta['end']}"
        )

    logger.info("\nBenchmark datasets created successfully!")
    logger.info(f"Available benchmarks: {benchmark.list_benchmarks()}")

    return forex_data, mtf_data


if __name__ == "__main__":
    # Create standard benchmarks
    create_standard_benchmarks()
