"""Shared fixtures and markers for SSVEP dataset smoke tests."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "download: marks tests that require downloading data (deselect with '-m \"not download\"')",
    )


def requires_download():
    """Skip marker for tests that need data download."""
    return pytest.mark.skipif(
        os.environ.get("SSVEP_TEST_DOWNLOAD", "0") != "1",
        reason="Set SSVEP_TEST_DOWNLOAD=1 to run download tests",
    )
