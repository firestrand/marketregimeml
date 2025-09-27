"""Model persistence handling.

Separated from base.py following Single Responsibility Principle.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import pickle
import warnings

import numpy as np

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class PersistenceHandler:
    """Handles model serialization and deserialization."""

    @staticmethod
    def save(
        data: Any,
        filepath: Path,
        format: Optional[str] = None,
        compression: Optional[str] = None,
    ) -> None:
        """Save data to file.

        Parameters
        ----------
        data : Any
            Data to save
        filepath : Path
            File path
        format : str, optional
            Format ('pickle', 'joblib', 'json')
        compression : str, optional
            Compression type for joblib
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = PersistenceHandler._detect_format(filepath)

        if format == "joblib":
            if not JOBLIB_AVAILABLE:
                warnings.warn("joblib not available, using pickle instead")
                format = "pickle"
            else:
                joblib.dump(data, filepath, compress=compression)
                return

        if format == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_data = PersistenceHandler._prepare_for_json(data)
            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=2)
        else:  # pickle
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    @staticmethod
    def _prepare_for_json(data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, dict):
            return {k: PersistenceHandler._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [PersistenceHandler._prepare_for_json(item) for item in data]
        else:
            return data

    @staticmethod
    def _detect_format(filepath: Path) -> str:
        """Detect format from file extension."""
        suffix = filepath.suffix.lower()
        if suffix in [".pkl", ".pickle"]:
            return "pickle"
        elif suffix in [".joblib", ".jbl"]:
            return "joblib"
        elif suffix == ".json":
            return "json"
        else:
            return "pickle"  # Default

    @staticmethod
    def _find_file(filepath: Path) -> Path:
        """Find file with any supported extension."""
        if filepath.exists():
            return filepath

        for ext in [".pkl", ".pickle", ".joblib", ".jbl", ".json"]:
            test_path = filepath.with_suffix(ext)
            if test_path.exists():
                return test_path

        raise FileNotFoundError(f"Model file not found: {filepath}")

    @staticmethod
    def load(filepath: Path, format: Optional[str] = None) -> Dict:
        """Load data from file.

        Parameters
        ----------
        filepath : Path
            File path
        format : str, optional
            Format to use

        Returns
        -------
        Dict
            Loaded data
        """
        filepath = PersistenceHandler._find_file(Path(filepath))

        if format is None:
            format = PersistenceHandler._detect_format(filepath)

        if format == "joblib":
            if not JOBLIB_AVAILABLE:
                format = "pickle"
            else:
                return joblib.load(filepath)

        if format == "json":
            with open(filepath, "r") as f:
                return json.load(f)
        else:  # pickle
            with open(filepath, "rb") as f:
                return pickle.load(f)