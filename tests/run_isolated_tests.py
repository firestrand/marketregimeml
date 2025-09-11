"""
Run tests in isolated subprocesses to avoid PyTorch reload issues.
"""

import subprocess
import sys
import json
import os


def run_test_in_subprocess(test_file, test_name=None):
    """Run a test in a subprocess and return coverage data."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "--cov=marketregimeml",
        "--cov-report=json",
        "--tb=short",
        "-q",
    ]

    if test_name:
        cmd.extend(["-k", test_name])

    # Run test in subprocess
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    if test_name:
        print(f"Test: {test_name}")
    print(f"{'='*60}")

    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr and "warning" not in result.stderr.lower():
        print("Errors:", result.stderr)

    # Try to read coverage data
    coverage_file = "coverage.json"
    if os.path.exists(coverage_file):
        with open(coverage_file, "r") as f:
            coverage_data = json.load(f)
            total = coverage_data.get("totals", {})
            print(f"\nCoverage Summary:")
            print(f"  Statements: {total.get('num_statements', 0)}")
            print(f"  Missing: {total.get('missing_lines', 0)}")
            print(f"  Coverage: {total.get('percent_covered', 0):.1f}%")
            return total.get("percent_covered", 0)

    return 0


def main():
    """Run all integration tests in isolation."""

    test_suites = [
        # Deep learning tests
        (
            "tests/test_deep_learning_integration_fixed.py",
            "test_lstm_basic_workflow",
        ),
        (
            "tests/test_deep_learning_integration_fixed.py",
            "test_transformer_basic_workflow",
        ),
        (
            "tests/test_deep_learning_integration_fixed.py",
            "test_cnn_lstm_basic_workflow",
        ),
        # ML model tests (if they exist)
        (
            "tests/test_ml_models_integration.py",
            "test_random_forest_full_workflow",
        ),
        ("tests/test_ml_models_integration.py", "test_xgboost_full_workflow"),
        # Ensemble tests
        (
            "tests/test_ensemble_integration.py",
            "test_ensemble_voting_strategy",
        ),
        (
            "tests/test_ensemble_integration.py",
            "test_ensemble_weighted_voting",
        ),
    ]

    coverages = []

    for test_file, test_name in test_suites:
        if os.path.exists(test_file):
            coverage = run_test_in_subprocess(test_file, test_name)
            coverages.append(coverage)
        else:
            print(f"Test file not found: {test_file}")

    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        print(f"\n{'='*60}")
        print(f"Overall Average Coverage: {avg_coverage:.1f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
