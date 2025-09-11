"""Benchmark datasets and evaluation framework for MarketRegimeML."""

from marketregimeml.benchmarks.benchmark_evaluation import BenchmarkEvaluator
from marketregimeml.benchmarks.create_benchmark_dataset import (
    BenchmarkDataset,
    create_standard_benchmarks,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkEvaluator",
    "create_standard_benchmarks",
]
