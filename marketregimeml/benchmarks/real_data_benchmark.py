#!/usr/bin/env python
"""Real market data benchmark system following SOLID, DRY, and KISS principles.

NO SYNTHETIC DATA - Only real market data from verified sources.
"""

import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.data.loaders import OANDADataLoader, KrakenDataLoader
from marketregimeml.data.loaders.alphavantage import AlphaVantageLoader


@dataclass
class BenchmarkResult:
    """Single benchmark result following Single Responsibility."""
    model_name: str
    dataset_name: str
    train_time: float
    inference_time: float  # ms per sample
    rqi: Optional[float]
    stability: Optional[float]
    n_samples: int
    n_features: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return asdict(self)
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class RealDataLoader:
    """Load only real market data - NO SYNTHETIC DATA."""
    
    def load_crypto(self, symbol: str = "BTC/USD", limit: int = 500) -> Optional[pd.DataFrame]:
        """Load cryptocurrency data from Kraken.
        
        Args:
            symbol: Crypto pair (BTC/USD, ETH/USD, etc.)
            limit: Number of daily samples
            
        Returns:
            DataFrame with features or None if failed
        """
        try:
            loader = KrakenDataLoader()
            
            end_date = datetime.now()
            # Kraken has good historical data
            start_date = end_date - timedelta(days=limit * 2)
            
            print(f"Loading {symbol} crypto data from Kraken")
            
            ohlcv = loader.fetch_ohlcv(
                symbol=symbol,
                timeframe="D",  # Daily
                start=start_date,
                end=end_date
            )
            
            # Trim to requested limit
            if len(ohlcv) > limit:
                ohlcv = ohlcv.iloc[-limit:]
            
            features = self._prepare_features(ohlcv)
            print(f"  Loaded {len(features)} {symbol} samples")
            return features
            
        except Exception as e:
            print(f"Failed to load crypto data for {symbol}: {e}")
            return None
    
    def load_spy(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Load SPY (S&P 500 ETF) data.
        
        Args:
            limit: Number of daily samples
            
        Returns:
            DataFrame with features or None if failed
        """
        try:
            loader = AlphaVantageLoader()
            
            end_date = datetime.now()
            # Add buffer for weekends/holidays
            start_date = end_date - timedelta(days=limit * 2)
            
            print(f"Loading SPY data from {start_date.date()} to {end_date.date()}")
            
            ohlcv = loader.fetch_ohlcv(
                symbol="SPY",
                timeframe="daily",
                start_date=start_date,
                end_date=end_date
            )
            
            # Trim to requested limit
            if len(ohlcv) > limit:
                ohlcv = ohlcv.iloc[-limit:]
            
            features = self._prepare_features(ohlcv)
            print(f"  Loaded {len(features)} SPY samples")
            return features
            
        except Exception as e:
            print(f"Failed to load SPY data: {e}")
            return None
    
    def load_qqq(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Load QQQ (NASDAQ-100 ETF) data.
        
        Args:
            limit: Number of daily samples
            
        Returns:
            DataFrame with features or None if failed
        """
        try:
            loader = AlphaVantageLoader()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit * 2)
            
            print(f"Loading QQQ data from {start_date.date()} to {end_date.date()}")
            
            ohlcv = loader.fetch_ohlcv(
                symbol="QQQ",
                timeframe="daily",
                start_date=start_date,
                end_date=end_date
            )
            
            if len(ohlcv) > limit:
                ohlcv = ohlcv.iloc[-limit:]
            
            features = self._prepare_features(ohlcv)
            print(f"  Loaded {len(features)} QQQ samples")
            return features
            
        except Exception as e:
            print(f"Failed to load QQQ data: {e}")
            return None
    
    def load_forex_data(
        self,
        symbol: str = "EUR_USD",
        timeframe: str = "D",
        limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """Load real FOREX data.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe (D, H1, M5, etc.)
            limit: Number of samples
            
        Returns:
            DataFrame with features or None if failed
        """
        try:
            loader = OANDADataLoader()
            
            end_date = datetime.now()
            if timeframe == "D":
                start_date = end_date - timedelta(days=limit * 2)
            elif timeframe == "H1":
                start_date = end_date - timedelta(hours=limit)
            else:  # M5
                start_date = end_date - timedelta(minutes=limit*5)
            
            print(f"Loading {symbol} data from {start_date.date()} to {end_date.date()}")
            
            ohlcv = loader.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(ohlcv) > limit:
                ohlcv = ohlcv.iloc[-limit:]
            
            features = self._prepare_features(ohlcv)
            print(f"  Loaded {len(features)} {symbol} samples")
            return features
            
        except Exception as e:
            print(f"Failed to load FOREX data: {e}")
            return None
    
    def load_stock_data(
        self,
        symbol: str = "AAPL",
        timeframe: str = "daily",
        limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """Load real stock data.
        
        Args:
            symbol: Stock ticker
            timeframe: Timeframe
            limit: Number of samples
            
        Returns:
            DataFrame with features or None if failed
        """
        try:
            loader = AlphaVantageLoader()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit * 2)
            
            print(f"Loading {symbol} data from {start_date.date()} to {end_date.date()}")
            
            ohlcv = loader.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(ohlcv) > limit:
                ohlcv = ohlcv.iloc[-limit:]
            
            features = self._prepare_features(ohlcv)
            print(f"  Loaded {len(features)} {symbol} samples")
            return features
            
        except Exception as e:
            print(f"Failed to load stock data for {symbol}: {e}")
            return None
    
    def _prepare_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from OHLCV data (DRY principle).
        
        Args:
            ohlcv: OHLCV DataFrame
            
        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame()
        
        # Returns
        features['returns'] = ohlcv['close'].pct_change()
        
        # Volatility measures
        # Parkinson volatility
        features['parkinson_vol'] = np.sqrt(
            np.log(ohlcv['high'] / ohlcv['low']) ** 2 / (4 * np.log(2))
        )
        
        # Garman-Klass volatility
        features['gk_vol'] = np.sqrt(
            0.5 * np.log(ohlcv['high'] / ohlcv['low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(ohlcv['close'] / ohlcv['open']) ** 2
        )
        
        # Rolling standard deviation
        features['rolling_vol'] = features['returns'].rolling(20).std()
        
        # Momentum indicators
        features['momentum_10'] = features['returns'].rolling(10).mean()
        features['momentum_20'] = features['returns'].rolling(20).mean()
        
        # RSI
        delta = ohlcv['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume (normalized by rolling mean)
        if 'volume' in ohlcv.columns and ohlcv['volume'].sum() > 0:
            vol_mean = ohlcv['volume'].rolling(20, min_periods=1).mean()
            features['volume_ratio'] = ohlcv['volume'] / (vol_mean + 1e-10)
        else:
            features['volume_ratio'] = 1.0
        
        # Price position within daily range
        features['price_position'] = (
            (ohlcv['close'] - ohlcv['low']) / 
            (ohlcv['high'] - ohlcv['low'] + 1e-10)
        )
        
        # ATR (Average True Range)
        high_low = ohlcv['high'] - ohlcv['low']
        high_close = np.abs(ohlcv['high'] - ohlcv['close'].shift())
        low_close = np.abs(ohlcv['low'] - ohlcv['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(14).mean()
        
        return features.dropna()


class ModelBenchmark:
    """Benchmark a single model following Single Responsibility."""
    
    def __init__(self, model, name: str):
        """Initialize benchmark.
        
        Args:
            model: Model instance
            name: Model name
        """
        self.model = model
        self.name = name
        self.metrics = RegimeMetrics()
    
    def run(self, data: pd.DataFrame, dataset_name: str) -> BenchmarkResult:
        """Run benchmark on dataset.
        
        Args:
            data: Feature DataFrame
            dataset_name: Name of dataset
            
        Returns:
            BenchmarkResult
        """
        # Special handling for MS-GARCH (needs returns only)
        if self.name.startswith("MS-GARCH"):
            benchmark_data = pd.DataFrame({'returns': data['returns']})
        else:
            benchmark_data = data
        
        # Training
        train_time = None
        try:
            start = time.time()
            self.model.fit(benchmark_data)
            train_time = time.time() - start
        except Exception as e:
            return BenchmarkResult(
                model_name=self.name,
                dataset_name=dataset_name,
                train_time=0,
                inference_time=0,
                rqi=None,
                stability=None,
                n_samples=len(data),
                n_features=benchmark_data.shape[1],
                metadata={'error': str(e)}
            )
        
        # Inference
        inference_time = None
        predictions = None
        try:
            start = time.time()
            predictions = self.model.predict(benchmark_data)
            total_time = time.time() - start
            inference_time = (total_time / len(benchmark_data)) * 1000  # ms
        except Exception as e:
            return BenchmarkResult(
                model_name=self.name,
                dataset_name=dataset_name,
                train_time=train_time,
                inference_time=0,
                rqi=None,
                stability=None,
                n_samples=len(data),
                n_features=benchmark_data.shape[1],
                metadata={'error': str(e)}
            )
        
        # Metrics
        rqi = None
        stability = None
        try:
            proba = self.model.predict_proba(benchmark_data)
            rqi = self.metrics.regime_quality_index(
                benchmark_data.values,
                predictions,
                proba
            )
            
            stability_metrics = self.metrics.regime_stability(predictions)
            stability = 1.0 - stability_metrics['transition_rate']
            
        except Exception as e:
            # RQI calculation failed, but benchmark completed
            pass
        
        return BenchmarkResult(
            model_name=self.name,
            dataset_name=dataset_name,
            train_time=train_time,
            inference_time=inference_time,
            rqi=rqi,
            stability=stability,
            n_samples=len(data),
            n_features=benchmark_data.shape[1]
        )


class RealDataBenchmarkRunner:
    """Benchmark runner for REAL DATA ONLY - NO SYNTHETICS."""
    
    def __init__(self):
        """Initialize runner."""
        self.models: Dict[str, Any] = {}
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.results: List[BenchmarkResult] = []
    
    def register_model(self, name: str, model):
        """Register a model for benchmarking.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
    
    def register_dataset(self, name: str, data: pd.DataFrame):
        """Register a dataset for benchmarking.
        
        Args:
            name: Dataset name
            data: Feature DataFrame
        """
        if data is not None and len(data) > 0:
            self.datasets[name] = data
            print(f"✓ Registered {name}: {len(data)} samples, {data.shape[1]} features")
        else:
            print(f"✗ Failed to register {name}: No data")
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all model-dataset combinations.
        
        Returns:
            List of BenchmarkResults
        """
        results = []
        
        total = len(self.models) * len(self.datasets)
        current = 0
        
        print("\n" + "=" * 60)
        print("RUNNING BENCHMARKS ON REAL MARKET DATA")
        print("=" * 60)
        
        for dataset_name, data in self.datasets.items():
            print(f"\n### Dataset: {dataset_name} ###")
            for model_name, model in self.models.items():
                current += 1
                print(f"\n[{current}/{total}] {model_name} on {dataset_name}")
                
                benchmark = ModelBenchmark(model, model_name)
                result = benchmark.run(data, dataset_name)
                results.append(result)
                
                # Print immediate feedback
                if result.rqi:
                    print(f"  ✓ Train: {result.train_time:.3f}s, "
                          f"Inference: {result.inference_time:.2f}ms, "
                          f"RQI: {result.rqi:.1f}, "
                          f"Stability: {result.stability:.1%}")
                else:
                    print(f"  ✗ Failed: {result.metadata.get('error', 'Unknown')}")
        
        self.results = results
        return results
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generate markdown report.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Markdown report string
        """
        report = []
        report.append("# Real Market Data Benchmark Report")
        report.append("\n**NO SYNTHETIC DATA - ONLY REAL MARKET DATA**")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Group by dataset
        datasets = {}
        for result in results:
            if result.dataset_name not in datasets:
                datasets[result.dataset_name] = []
            datasets[result.dataset_name].append(result)
        
        for dataset_name, dataset_results in datasets.items():
            report.append(f"\n## {dataset_name}")
            report.append("\n| Model | Train Time | Inference | RQI | Stability |")
            report.append("|-------|------------|-----------|-----|-----------|")
            
            # Sort by RQI (best first)
            dataset_results.sort(
                key=lambda x: x.rqi if x.rqi else 0,
                reverse=True
            )
            
            for result in dataset_results:
                train = f"{result.train_time:.3f}s"
                infer = f"{result.inference_time:.2f}ms"
                rqi = f"{result.rqi:.1f}" if result.rqi else "N/A"
                stab = f"{result.stability:.1%}" if result.stability else "N/A"
                
                # Highlight top performer
                if result == dataset_results[0] and result.rqi:
                    report.append(f"| **{result.model_name}** | **{train}** | **{infer}** | **{rqi}** | **{stab}** |")
                else:
                    report.append(f"| {result.model_name} | {train} | {infer} | {rqi} | {stab} |")
        
        # Overall summary
        report.append("\n## Overall Performance Summary")
        
        # Calculate average RQI per model across all datasets
        model_avg_rqi = {}
        for result in results:
            if result.rqi:
                if result.model_name not in model_avg_rqi:
                    model_avg_rqi[result.model_name] = []
                model_avg_rqi[result.model_name].append(result.rqi)
        
        avg_scores = [(name, np.mean(scores)) for name, scores in model_avg_rqi.items()]
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        
        report.append("\n### Average RQI Across All Real Datasets")
        report.append("\n| Rank | Model | Average RQI |")
        report.append("|------|-------|-------------|")
        for i, (name, avg_rqi) in enumerate(avg_scores, 1):
            report.append(f"| {i} | {name} | {avg_rqi:.1f} |")
        
        # Best model per dataset
        report.append("\n### Best Model Per Dataset")
        for dataset_name, dataset_results in datasets.items():
            best = max((r for r in dataset_results if r.rqi), 
                      key=lambda x: x.rqi, 
                      default=None)
            if best:
                report.append(f"- **{dataset_name}**: {best.model_name} (RQI: {best.rqi:.1f})")
        
        # Speed analysis
        report.append("\n### Speed Analysis")
        
        fastest_train = min(
            (r for r in results if r.train_time),
            key=lambda x: x.train_time,
            default=None
        )
        if fastest_train:
            report.append(f"- **Fastest Training**: {fastest_train.model_name} ({fastest_train.train_time:.3f}s)")
        
        fastest_infer = min(
            (r for r in results if r.inference_time),
            key=lambda x: x.inference_time,
            default=None
        )
        if fastest_infer:
            report.append(f"- **Fastest Inference**: {fastest_infer.model_name} ({fastest_infer.inference_time:.2f}ms)")
        
        return "\n".join(report)
    
    def save_results(self, results: List[BenchmarkResult], filename: str):
        """Save results to CSV.
        
        Args:
            results: List of benchmark results
            filename: Output filename
        """
        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")


def run_real_data_benchmark():
    """Run comprehensive benchmark on REAL market data only."""
    print("=" * 60)
    print("REAL MARKET DATA BENCHMARK")
    print("NO SYNTHETIC DATA")
    print("=" * 60)
    
    # Initialize components
    runner = RealDataBenchmarkRunner()
    loader = RealDataLoader()
    
    # Load REAL datasets
    print("\nLoading REAL market data...")
    
    # Major indices
    spy_data = loader.load_spy(500)
    if spy_data is not None:
        runner.register_dataset("SPY (S&P 500)", spy_data)
    
    qqq_data = loader.load_qqq(500)
    if qqq_data is not None:
        runner.register_dataset("QQQ (NASDAQ-100)", qqq_data)
    
    # Individual stocks
    aapl_data = loader.load_stock_data("AAPL", "daily", 500)
    if aapl_data is not None:
        runner.register_dataset("AAPL", aapl_data)
    
    msft_data = loader.load_stock_data("MSFT", "daily", 500)
    if msft_data is not None:
        runner.register_dataset("MSFT", msft_data)
    
    # FOREX
    eurusd_data = loader.load_forex_data("EUR_USD", "D", 500)
    if eurusd_data is not None:
        runner.register_dataset("EUR/USD", eurusd_data)
    
    gbpusd_data = loader.load_forex_data("GBP_USD", "D", 500)
    if gbpusd_data is not None:
        runner.register_dataset("GBP/USD", gbpusd_data)
    
    # CRYPTO
    btc_data = loader.load_crypto("BTC/USD", 500)
    if btc_data is not None:
        runner.register_dataset("BTC/USD", btc_data)
    
    eth_data = loader.load_crypto("ETH/USD", 500)
    if eth_data is not None:
        runner.register_dataset("ETH/USD", eth_data)
    
    print(f"\nLoaded {len(runner.datasets)} real market datasets")
    
    # Register all models
    print("\nRegistering models...")
    
    from marketregimeml.models import (
        GMMRegimeDetector,
        HMMRegimeDetector,
        GARCHRegimeDetector,
        RandomForestRegimeClassifier,
        XGBoostRegimeClassifier,
        SVMRegimeClassifier,
        MSGARCHRegimeDetector,
        EnsembleRegimeDetector
    )
    from marketregimeml.models.deep_learning import LSTMRegimeDetector
    
    # Using n_regimes=3 (optimal for ensemble models on real data)
    # Note: While n_regimes=5 showed benefits in earlier synthetic tests,
    # real market data benchmarks show n_regimes=3 provides better RQI scores
    models = {
        "GMM": GMMRegimeDetector(n_regimes=3, random_state=42),
        "HMM": HMMRegimeDetector(n_regimes=3, random_state=42),
        "GARCH": GARCHRegimeDetector(n_regimes=3, random_state=42),
        "Random Forest": RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=100, random_state=42
        ),
        "XGBoost": XGBoostRegimeClassifier(
            n_regimes=3, n_estimators=100, random_state=42
        ),
        "SVM (RBF)": SVMRegimeClassifier(
            n_regimes=3, kernel='rbf', probability=True, random_state=42
        ),
        "SVM (Linear)": SVMRegimeClassifier(
            n_regimes=3, kernel='linear', probability=True, random_state=42
        ),
        "MS-GARCH": MSGARCHRegimeDetector(n_regimes=2, random_state=42),  # MS-GARCH works best with 2 regimes
        "Ensemble (Top3)": EnsembleRegimeDetector(
            models=[
                XGBoostRegimeClassifier(n_regimes=3, n_estimators=100, random_state=42),
                SVMRegimeClassifier(n_regimes=3, kernel='rbf', probability=True, random_state=42),
                RandomForestRegimeClassifier(n_regimes=3, n_estimators=100, random_state=42)
            ],
            n_regimes=3,
            strategy='voting'
        ),
        "Ensemble (Fast SVM)": EnsembleRegimeDetector(
            models=[
                SVMRegimeClassifier(n_regimes=3, kernel='rbf', probability=True, random_state=42),
                SVMRegimeClassifier(n_regimes=3, kernel='linear', probability=True, random_state=42)
            ],
            n_regimes=3,
            strategy='voting'
        ),
        "LSTM": LSTMRegimeDetector(
            n_regimes=3,
            hidden_size=64,
            n_layers=2,
            epochs=10,
            batch_size=32,
            device='cpu'
        )
    }
    
    for name, model in models.items():
        runner.register_model(name, model)
    
    print(f"Registered {len(runner.models)} models")
    
    # Run benchmarks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = runner.run_all()
    
    # Generate report
    report = runner.generate_report(results)
    print("\n" + "=" * 60)
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(results, f"real_data_benchmark_{timestamp}.csv")
    
    # Save report
    with open(f"real_data_benchmark_{timestamp}.md", 'w') as f:
        f.write(report)
    print(f"Report saved to real_data_benchmark_{timestamp}.md")
    
    return results


if __name__ == "__main__":
    run_real_data_benchmark()