"""Benchmark datasets and evaluation framework for MarketRegimeML.

Keep imports lightweight to avoid pulling optional dependencies when
consuming specific submodules (e.g., real_data_benchmark).
"""

try:
    from marketregimeml.benchmarks.benchmark_evaluation import BenchmarkEvaluator
except Exception:
    BenchmarkEvaluator = None  # type: ignore

try:
    from marketregimeml.benchmarks.create_benchmark_dataset import (
        BenchmarkDataset,
        create_standard_benchmarks,
    )
except Exception:
    BenchmarkDataset = None  # type: ignore
    create_standard_benchmarks = None  # type: ignore

__all__ = [
    "BenchmarkDataset",
    "BenchmarkEvaluator",
    "create_standard_benchmarks",
]
