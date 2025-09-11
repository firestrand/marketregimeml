"""Tests for logging utilities."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from marketregimeml.utils.logging import setup_logging, get_logger


class TestLogging:
    """Test suite for logging utilities."""

    def teardown_method(self):
        """Clean up after each test."""
        # Reset root logger
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    @patch("marketregimeml.utils.logging.config_loader.get_logging_config")
    def test_setup_logging_defaults(self, mock_get_config):
        """Test setup_logging with default parameters."""
        mock_get_config.side_effect = ValueError("No config")
        setup_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_debug_level(self):
        """Test setup_logging with DEBUG level."""
        setup_logging(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_file(self):
        """Test setup_logging with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file=str(log_file))

            root_logger = logging.getLogger()
            # Should have both console and file handlers
            assert len(root_logger.handlers) == 2

            # Test that file is created
            test_logger = get_logger("test")
            test_logger.info("Test message")

            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content

    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format."""
        custom_format = "%(levelname)s: %(message)s"
        setup_logging(format_string=custom_format)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert handler.formatter._fmt == custom_format

    def test_get_logger(self):
        """Test get_logger function."""
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module2")
        logger3 = get_logger("test.module1")

        assert logger1.name == "test.module1"
        assert logger2.name == "test.module2"
        assert logger1 is logger3  # Same logger instance

    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs" / "nested" / "test.log"
            assert not log_file.parent.exists()

            setup_logging(log_file=str(log_file))

            assert log_file.parent.exists()

    @patch("marketregimeml.utils.logging.config_loader.get_logging_config")
    def test_multiple_setup_calls_replace_handlers(self, mock_get_config):
        """Test that multiple setup_logging calls don't duplicate handlers."""
        mock_get_config.side_effect = ValueError("No config")
        setup_logging(level="INFO")
        setup_logging(level="DEBUG")
        setup_logging(level="WARNING")

        root_logger = logging.getLogger()
        # Should still have only one handler
        assert len(root_logger.handlers) == 1
        assert root_logger.level == logging.WARNING
