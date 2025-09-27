"""Feature data validation following Single Responsibility Principle."""

from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureValidator:
    """Validates feature data and input parameters.

    Single Responsibility: Data validation and quality checks.
    """

    def __init__(self):
        """Initialize feature validator."""
        logger.debug("FeatureValidator initialized")

    def validate_input_data(self, data: Any) -> bool:
        """Validate input data format and structure.

        Args:
            data: Input data to validate

        Returns:
            True if data is valid, False otherwise

        Raises:
            ValueError: If data validation fails
        """
        try:
            # Check if data is DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data must be pandas DataFrame, got {type(data)}")

            # Check if DataFrame is empty
            if data.empty:
                raise ValueError("Input DataFrame is empty")

            # Check for minimum required columns (OHLCV structure)
            required_base_cols = ["close"]  # Minimum requirement
            if not any(col in data.columns for col in required_base_cols):
                logger.warning(
                    f"No standard price columns found. Available: {list(data.columns)}"
                )

            # Check data types
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in data")

            # Check for reasonable data size
            min_rows = 10  # Minimum for meaningful calculations
            if len(data) < min_rows:
                logger.warning(
                    f"Data has only {len(data)} rows, minimum {min_rows} recommended"
                )

            logger.debug(
                f"Input data validated: {data.shape} shape, {len(numeric_cols)} numeric columns"
            )
            return True

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise

    def validate_ohlc_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate OHLC data structure and relationships.

        Args:
            data: DataFrame with potential OHLC columns

        Returns:
            Dictionary with validation results and available columns
        """
        validation_result = {
            "has_ohlc": False,
            "available_columns": list(data.columns),
            "missing_columns": [],
            "data_issues": [],
        }

        # Check for OHLC columns
        ohlc_cols = ["open", "high", "low", "close"]
        available_ohlc = [col for col in ohlc_cols if col in data.columns]
        missing_ohlc = [col for col in ohlc_cols if col not in data.columns]

        validation_result["available_ohlc"] = available_ohlc
        validation_result["missing_columns"] = missing_ohlc
        validation_result["has_ohlc"] = (
            len(available_ohlc) >= 3
        )  # At least 3 of 4 columns

        # Validate OHLC relationships if we have the data
        if len(available_ohlc) >= 3:
            try:
                if all(col in data.columns for col in ["open", "high", "low", "close"]):
                    # Check OHLC constraints: High >= Low, High >= Open/Close, Low <= Open/Close
                    invalid_hl = (data["high"] < data["low"]).sum()
                    invalid_ho = (data["high"] < data["open"]).sum()
                    invalid_hc = (data["high"] < data["close"]).sum()
                    invalid_lo = (data["low"] > data["open"]).sum()
                    invalid_lc = (data["low"] > data["close"]).sum()

                    total_violations = (
                        invalid_hl + invalid_ho + invalid_hc + invalid_lo + invalid_lc
                    )
                    if total_violations > 0:
                        validation_result["data_issues"].append(
                            f"OHLC constraint violations: {total_violations} rows"
                        )
            except Exception as e:
                validation_result["data_issues"].append(f"OHLC validation error: {e}")

        return validation_result

    def validate_computed_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate computed features for quality and consistency.

        Args:
            features: Computed features DataFrame

        Returns:
            Dictionary with validation results and quality metrics
        """
        validation_result = self._initialize_validation_result(features)

        try:
            if features.empty:
                validation_result["is_valid"] = False
                validation_result["issues"].append("Features DataFrame is empty")
                return validation_result

            # Run all validation checks
            self._check_nan_columns(features, validation_result)
            self._check_infinite_values(features, validation_result)
            self._check_constant_columns(features, validation_result)

            # Calculate overall quality
            self._calculate_quality_score(features, validation_result)
            self._determine_validity(features, validation_result)

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Validation error: {e}")
            logger.error(f"Feature validation failed: {e}")

        return validation_result

    def validate_feature_types(self, requested_types: List[str]) -> Dict[str, Any]:
        """Validate requested feature types.

        Args:
            requested_types: List of requested feature type names

        Returns:
            Dictionary with validation results
        """
        valid_types = {"volatility", "entropy", "statistical", "technical"}

        validation_result = {
            "valid_types": [],
            "invalid_types": [],
            "all_valid": True,
        }

        for feature_type in requested_types:
            if feature_type in valid_types:
                validation_result["valid_types"].append(feature_type)
            else:
                validation_result["invalid_types"].append(feature_type)
                validation_result["all_valid"] = False

        if validation_result["invalid_types"]:
            logger.warning(
                f"Invalid feature types requested: {validation_result['invalid_types']}"
            )
            logger.info(f"Valid types are: {valid_types}")

        return validation_result

    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature computation parameters.

        Args:
            params: Dictionary of parameters to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid_params": {},
            "invalid_params": {},
            "warnings": [],
        }

        for param_name, param_value in params.items():
            self._validate_single_parameter(param_name, param_value, validation_result)

        return validation_result

    def _validate_single_parameter(
        self, param_name: str, param_value: Any, result: Dict[str, Any]
    ) -> None:
        """Validate a single parameter and update result dictionary."""
        try:
            validator = self._get_parameter_validator(param_name)
            is_valid, error_msg = validator(param_value)

            if is_valid:
                result["valid_params"][param_name] = param_value
            else:
                result["invalid_params"][param_name] = error_msg

            # Add warning for unknown parameters
            if validator == self._validate_unknown:
                result["warnings"].append(f"Unknown parameter: {param_name}")

        except Exception as e:
            result["invalid_params"][param_name] = f"Validation error: {e}"

    def _get_parameter_validator(self, param_name: str):
        """Get appropriate validator for parameter."""
        if param_name.endswith("_window") or param_name == "window":
            return self._validate_window
        elif param_name in ["alpha", "beta", "r", "threshold"]:
            return self._validate_fraction
        elif param_name == "bins":
            return self._validate_bins
        else:
            return self._validate_unknown

    def _validate_window(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate window parameter."""
        if not isinstance(value, int) or value < 2:
            return False, f"Window must be integer >= 2, got {value}"
        return True, None

    def _validate_fraction(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate fraction/probability parameter."""
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            return False, f"Parameter must be float [0,1], got {value}"
        return True, None

    def _validate_bins(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate bins parameter."""
        if not isinstance(value, int) or value < 2:
            return False, f"Bins must be integer >= 2, got {value}"
        return True, None

    def _validate_unknown(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Pass through unknown parameters."""
        return True, None

    def _initialize_validation_result(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Initialize validation result dictionary."""
        return {
            "is_valid": True,
            "feature_count": len(features.columns),
            "row_count": len(features),
            "issues": [],
            "quality_metrics": {},
        }

    def _check_nan_columns(
        self, features: pd.DataFrame, result: Dict[str, Any]
    ) -> None:
        """Check for NaN values in features."""
        # Check for all-NaN columns
        nan_columns = features.columns[features.isna().all()].tolist()
        if nan_columns:
            result["issues"].append(f"All-NaN columns: {len(nan_columns)}")
            result["quality_metrics"]["all_nan_columns"] = nan_columns

        # Check for excessive NaN values
        nan_percentages = features.isna().sum() / len(features) * 100
        high_nan_cols = nan_percentages[nan_percentages > 50].index.tolist()
        if high_nan_cols:
            result["issues"].append(f"High NaN columns (>50%): {len(high_nan_cols)}")
            result["quality_metrics"]["high_nan_columns"] = high_nan_cols

    def _check_infinite_values(
        self, features: pd.DataFrame, result: Dict[str, Any]
    ) -> None:
        """Check for infinite values in numeric columns."""
        inf_columns = []
        for col in features.select_dtypes(include=[np.number]).columns:
            if np.isinf(features[col]).any():
                inf_columns.append(col)

        if inf_columns:
            result["issues"].append(f"Columns with infinite values: {len(inf_columns)}")
            result["quality_metrics"]["infinite_columns"] = inf_columns

    def _check_constant_columns(
        self, features: pd.DataFrame, result: Dict[str, Any]
    ) -> None:
        """Check for constant (no variance) columns."""
        constant_columns = []
        for col in features.select_dtypes(include=[np.number]).columns:
            if features[col].dropna().std() == 0:
                constant_columns.append(col)

        if constant_columns:
            result["issues"].append(f"Constant columns: {len(constant_columns)}")
            result["quality_metrics"]["constant_columns"] = constant_columns

    def _calculate_quality_score(
        self, features: pd.DataFrame, result: Dict[str, Any]
    ) -> None:
        """Calculate overall quality score for features."""
        metrics = result["quality_metrics"]
        nan_cols = len(metrics.get("all_nan_columns", []))
        high_nan_cols = len(metrics.get("high_nan_columns", []))
        inf_cols = len(metrics.get("infinite_columns", []))
        const_cols = len(metrics.get("constant_columns", []))

        total_issues = nan_cols + high_nan_cols + inf_cols + const_cols
        quality_score = max(0, 1 - (total_issues / len(features.columns)))
        result["quality_metrics"]["overall_score"] = quality_score

    def _determine_validity(
        self, features: pd.DataFrame, result: Dict[str, Any]
    ) -> None:
        """Determine if features are valid based on critical issues."""
        metrics = result["quality_metrics"]
        nan_cols = len(metrics.get("all_nan_columns", []))
        inf_cols = len(metrics.get("infinite_columns", []))

        critical_issues = nan_cols + inf_cols
        if (
            critical_issues > len(features.columns) * 0.2
        ):  # More than 20% critical issues
            result["is_valid"] = False
