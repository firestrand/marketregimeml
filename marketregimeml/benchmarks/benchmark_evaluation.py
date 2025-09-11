"""Benchmark evaluation framework for regime detection models.

This module provides tools to evaluate and compare regime detection
models using standardized benchmark datasets.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from marketregimeml.evaluation import RegimeMetrics
from marketregimeml.models import (
    HMMRegimeDetector,
    GMMRegimeDetector,
    GARCHRegimeDetector,
    AdaptiveOrchestrator,
)
from marketregimeml.utils.logging import get_logger
from marketregimeml.visualization import RegimePlotter

logger = get_logger(__name__)


class BenchmarkEvaluator:
    """Evaluates regime detection models on benchmark datasets."""

    def __init__(self, benchmark_data: Dict[str, pd.DataFrame]):
        """Initialize benchmark evaluator.

        Args:
            benchmark_data: Benchmark dataset dictionary
        """
        self.benchmark_data = benchmark_data
        self.metrics = RegimeMetrics()
        self.plotter = RegimePlotter()
        self.results = {}

    def evaluate_model(
        self, model, data_key: str, n_regimes: int = 3, test_split: float = 0.2
    ) -> Dict[str, float]:
        """Evaluate a single model on benchmark data.

        Args:
            model: Regime detection model
            data_key: Key for benchmark data
            n_regimes: Number of regimes
            test_split: Fraction of data for testing

        Returns:
            Dictionary of evaluation metrics
        """
        if data_key not in self.benchmark_data:
            raise KeyError(f"Data key {data_key} not found in benchmark data")

        data = self.benchmark_data[data_key]
        features = data["features"]

        # Split data
        split_idx = int(len(features) * (1 - test_split))
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]

        # Train model - use only numeric features
        numeric_features = train_features.select_dtypes(include=[np.number])
        if "true_regime" in numeric_features.columns:
            numeric_features = numeric_features.drop(columns=["true_regime"])

        # Fill any NaN values
        numeric_features = numeric_features.fillna(method="ffill").fillna(0)

        start_time = time.time()
        model.fit(numeric_features)
        train_time = time.time() - start_time

        # Predict on test set - use numeric features
        test_numeric = test_features.select_dtypes(include=[np.number])
        if "true_regime" in test_numeric.columns:
            test_numeric = test_numeric.drop(columns=["true_regime"])
        test_numeric = test_numeric.fillna(method="ffill").fillna(0)

        start_time = time.time()
        test_regimes = model.predict(test_numeric)
        test_proba = model.predict_proba(test_numeric)
        inference_time = time.time() - start_time

        # Calculate metrics
        metrics = {
            "train_time": train_time,
            "inference_time": inference_time,
            "inference_speed": len(test_features) / inference_time,
        }

        # Regime quality metrics - calculate individual metrics
        quality_metrics = {}

        # Stability metrics
        stability_dict = self.metrics.regime_stability(test_regimes)
        quality_metrics.update(stability_dict)

        # Distribution metrics
        dist_dict = self.metrics.regime_distribution(test_regimes)
        quality_metrics["regime_balance"] = dist_dict.get("balance", 0)

        # Confidence metrics if probabilities available
        if test_proba is not None:
            conf_dict = self.metrics.confidence_metrics(test_proba)
            quality_metrics.update(conf_dict)

        # Overall quality index
        quality_idx = self.metrics.regime_quality_index(
            test_numeric.values, test_regimes, test_proba
        )
        quality_metrics["quality_index"] = quality_idx

        metrics.update(quality_metrics)

        # If true regimes available, calculate supervised metrics
        if "true_regime" in test_features.columns:
            true_regimes = test_features["true_regime"].values
            metrics["adjusted_rand_score"] = adjusted_rand_score(
                true_regimes, test_regimes
            )

        # Clustering metrics
        if len(np.unique(test_regimes)) > 1:
            metrics["silhouette_score"] = silhouette_score(
                test_features.select_dtypes(include=[np.number]).fillna(0),
                test_regimes,
            )
            metrics["calinski_harabasz"] = calinski_harabasz_score(
                test_features.select_dtypes(include=[np.number]).fillna(0),
                test_regimes,
            )
            metrics["davies_bouldin"] = davies_bouldin_score(
                test_features.select_dtypes(include=[np.number]).fillna(0),
                test_regimes,
            )

        return metrics

    def compare_models(
        self,
        models: Dict[str, object],
        data_keys: List[str] = None,
        n_regimes: int = 3,
    ) -> pd.DataFrame:
        """Compare multiple models on benchmark datasets.

        Args:
            models: Dictionary of models to compare
            data_keys: Keys for benchmark data to use
            n_regimes: Number of regimes

        Returns:
            DataFrame with comparison results
        """
        if data_keys is None:
            data_keys = list(self.benchmark_data.keys())

        results = []

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")

            for data_key in data_keys:
                logger.info(f"  on {data_key}")

                try:
                    metrics = self.evaluate_model(model, data_key, n_regimes)
                    metrics["model"] = model_name
                    metrics["dataset"] = data_key
                    results.append(metrics)

                except Exception as e:
                    logger.error(
                        f"Failed to evaluate {model_name} on {data_key}: {e}"
                    )

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)

        # Calculate rankings
        ranking_metrics = [
            "persistence",
            "stability",
            "separation",
            "silhouette_score",
            "calinski_harabasz",
        ]

        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                # Higher is better for these metrics
                comparison_df[f"{metric}_rank"] = comparison_df.groupby(
                    "dataset"
                )[metric].rank(ascending=False)

        # Davies-Bouldin: lower is better
        if "davies_bouldin" in comparison_df.columns:
            comparison_df["davies_bouldin_rank"] = comparison_df.groupby(
                "dataset"
            )["davies_bouldin"].rank(ascending=True)

        # Calculate average rank
        rank_cols = [
            col for col in comparison_df.columns if col.endswith("_rank")
        ]
        if rank_cols:
            comparison_df["avg_rank"] = comparison_df[rank_cols].mean(axis=1)

        self.results = comparison_df

        return comparison_df

    def create_ensemble_benchmark(
        self, base_models: Dict[str, object], data_key: str, n_regimes: int = 3
    ) -> Dict[str, float]:
        """Benchmark adaptive ensemble model.

        Args:
            base_models: Dictionary of base models
            data_key: Key for benchmark data
            n_regimes: Number of regimes

        Returns:
            Ensemble performance metrics
        """
        # Create adaptive orchestrator
        orchestrator = AdaptiveOrchestrator(
            models=base_models,
            selection_strategy="thompson_sampling",
            context_aware=True,
            use_fuzzy_transitions=True,
        )

        # Evaluate
        metrics = self.evaluate_model(orchestrator, data_key, n_regimes)
        metrics["model"] = "AdaptiveEnsemble"

        # Get model selection statistics
        selection_stats = orchestrator.get_performance_metrics()
        metrics["model_selection"] = selection_stats

        return metrics

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate benchmark evaluation report.

        Args:
            save_path: Path to save report

        Returns:
            Report as string
        """
        if self.results.empty:
            return "No results available. Run compare_models first."

        report = []
        report.append("=" * 80)
        report.append("REGIME DETECTION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall best model
        if "avg_rank" in self.results.columns:
            best_model_idx = self.results["avg_rank"].idxmin()
            best_model = self.results.loc[best_model_idx]

            report.append("BEST OVERALL MODEL")
            report.append("-" * 40)
            report.append(f"Model: {best_model['model']}")
            report.append(f"Average Rank: {best_model['avg_rank']:.2f}")
            report.append("")

        # Performance by dataset
        for dataset in self.results["dataset"].unique():
            dataset_results = self.results[self.results["dataset"] == dataset]

            report.append(f"DATASET: {dataset}")
            report.append("-" * 40)

            # Top models for this dataset
            if "avg_rank" in dataset_results.columns:
                top_models = dataset_results.nsmallest(3, "avg_rank")

                for idx, row in top_models.iterrows():
                    report.append(f"  {row['model']}:")
                    report.append(
                        f"    - Persistence: {row.get('persistence', 0):.3f}"
                    )
                    report.append(
                        f"    - Stability: {row.get('stability', 0):.3f}"
                    )
                    report.append(
                        f"    - Train Time: {row.get('train_time', 0):.2f}s"
                    )
                    report.append(
                        f"    - Inference Speed: {row.get('inference_speed', 0):.0f} samples/s"
                    )

            report.append("")

        # Timing analysis
        report.append("COMPUTATIONAL PERFORMANCE")
        report.append("-" * 40)

        timing_summary = self.results.groupby("model")[
            ["train_time", "inference_speed"]
        ].mean()
        for model, row in timing_summary.iterrows():
            report.append(f"  {model}:")
            report.append(f"    - Avg Train Time: {row['train_time']:.2f}s")
            report.append(
                f"    - Avg Inference Speed: {row['inference_speed']:.0f} samples/s"
            )

        report.append("")
        report.append("=" * 80)

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_str)
            logger.info(f"Report saved to {save_path}")

        return report_str

    def plot_comparison(
        self, metric: str = "persistence", save_path: Optional[str] = None
    ):
        """Plot model comparison for a specific metric.

        Args:
            metric: Metric to plot
            save_path: Path to save plot
        """
        if self.results.empty:
            logger.warning("No results to plot. Run compare_models first.")
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Pivot data for plotting
        pivot_data = self.results.pivot(
            index="model", columns="dataset", values=metric
        )

        pivot_data.plot(kind="bar", ax=ax)

        ax.set_title(f"Model Comparison: {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.legend(title="Dataset")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")

        plt.show()


def run_standard_benchmark():
    """Run standard benchmark evaluation."""

    from marketregimeml.benchmarks.create_benchmark_dataset import (
        BenchmarkDataset,
    )

    # Load benchmark data
    benchmark = BenchmarkDataset()

    try:
        forex_data = benchmark.load_benchmark("forex_benchmark_30d")
    except FileNotFoundError:
        logger.info("Creating benchmark dataset first...")
        from marketregimeml.benchmarks.create_benchmark_dataset import (
            create_standard_benchmarks,
        )

        forex_data, _ = create_standard_benchmarks()

    # Initialize models
    models = {
        "HMM": HMMRegimeDetector(n_regimes=3),
        "GMM": GMMRegimeDetector(n_components=3),
        "GARCH": GARCHRegimeDetector(n_regimes=3),
    }

    # Create evaluator
    evaluator = BenchmarkEvaluator(forex_data)

    # Compare models
    logger.info("Comparing models on benchmark datasets...")
    comparison = evaluator.compare_models(models, data_keys=["EUR_USD"])

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(comparison.to_string())

    # Generate report
    report = evaluator.generate_report("benchmark_report.txt")
    print("\n" + report)

    # Test ensemble
    logger.info("\nTesting adaptive ensemble...")
    ensemble_metrics = evaluator.create_ensemble_benchmark(models, "EUR_USD")

    print("\nADAPTIVE ENSEMBLE RESULTS:")
    for key, value in ensemble_metrics.items():
        if not isinstance(value, dict):
            print(
                f"  {key}: {value:.3f}"
                if isinstance(value, float)
                else f"  {key}: {value}"
            )

    return evaluator


if __name__ == "__main__":
    # Run standard benchmark
    evaluator = run_standard_benchmark()
