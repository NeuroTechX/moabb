"""Tests for the Emissions class in evaluations/utils.py."""

from unittest.mock import MagicMock, patch

import pytest

from moabb.evaluations.utils import Emissions


try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker  # noqa

    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


def test_default_config():
    """Test Emissions initialization with default config (None)."""
    emissions = Emissions()

    # Check default configuration is set
    assert emissions.codecarbon_config == dict(save_to_file=False, log_level="error")
    # Check codecarbon_offline is set to False for default config
    assert emissions.codecarbon_offline is False


def test_custom_config_online_mode():
    """Test Emissions initialization with custom config (online mode)."""
    custom_config = {
        "save_to_file": True,
        "log_level": "info",
        "project_name": "test_project",
    }
    emissions = Emissions(codecarbon_config=custom_config)

    # Check custom configuration is set
    assert emissions.codecarbon_config == custom_config
    # Check codecarbon_offline is False when no offline params are present
    assert emissions.codecarbon_offline is False


@pytest.mark.parametrize(
    "offline_param,param_value",
    [
        ("country_iso_code", "USA"),
        ("region", "us-west-1"),
        ("cloud_provider", "aws"),
        ("cloud_region", "us-west-1"),
        ("country_2letter_iso_code", "US"),
    ],
)
def test_custom_config_offline_mode(offline_param, param_value):
    """Test Emissions initialization with various offline parameters."""
    custom_config = {
        "save_to_file": False,
        "log_level": "error",
        offline_param: param_value,
    }
    emissions = Emissions(codecarbon_config=custom_config)

    # Check custom configuration is set
    assert emissions.codecarbon_config == custom_config
    # Check codecarbon_offline is True when offline param is present
    assert emissions.codecarbon_offline is True


def test_custom_config_offline_mode_multiple_params():
    """Test Emissions initialization with multiple offline parameters."""
    custom_config = {
        "save_to_file": True,
        "log_level": "info",
        "country_iso_code": "USA",
        "region": "california",
        "cloud_provider": "aws",
    }
    emissions = Emissions(codecarbon_config=custom_config)

    assert emissions.codecarbon_config == custom_config
    assert emissions.codecarbon_offline is True


@pytest.mark.skipif(not CODECARBON_AVAILABLE, reason="codecarbon not installed")
@patch("moabb.evaluations.utils.EmissionsTracker")
def test_create_tracker_default_config(mock_emissions_tracker):
    """Test create_tracker with default config uses EmissionsTracker."""
    mock_tracker = MagicMock()
    mock_emissions_tracker.return_value = mock_tracker

    emissions = Emissions()
    tracker = emissions.create_tracker()

    # Verify EmissionsTracker was called with correct config
    mock_emissions_tracker.assert_called_once_with(save_to_file=False, log_level="error")
    assert tracker == mock_tracker


@pytest.mark.skipif(not CODECARBON_AVAILABLE, reason="codecarbon not installed")
@patch("moabb.evaluations.utils.EmissionsTracker")
def test_create_tracker_online_mode(mock_emissions_tracker):
    """Test create_tracker with online config uses EmissionsTracker."""
    mock_tracker = MagicMock()
    mock_emissions_tracker.return_value = mock_tracker

    custom_config = {"save_to_file": True, "log_level": "info"}
    emissions = Emissions(codecarbon_config=custom_config)
    tracker = emissions.create_tracker()

    # Verify EmissionsTracker was called with custom config
    mock_emissions_tracker.assert_called_once_with(save_to_file=True, log_level="info")
    assert tracker == mock_tracker


@pytest.mark.skipif(not CODECARBON_AVAILABLE, reason="codecarbon not installed")
@patch("moabb.evaluations.utils.OfflineEmissionsTracker")
def test_create_tracker_offline_mode(mock_offline_emissions_tracker):
    """Test create_tracker with offline config uses OfflineEmissionsTracker."""
    mock_tracker = MagicMock()
    mock_offline_emissions_tracker.return_value = mock_tracker

    custom_config = {
        "save_to_file": False,
        "log_level": "error",
        "country_iso_code": "USA",
    }
    emissions = Emissions(codecarbon_config=custom_config)
    tracker = emissions.create_tracker()

    # Verify OfflineEmissionsTracker was called with custom config
    mock_offline_emissions_tracker.assert_called_once_with(
        save_to_file=False, log_level="error", country_iso_code="USA"
    )
    assert tracker == mock_tracker


@pytest.mark.parametrize(
    "config,expected_offline",
    [
        (None, False),
        ({}, False),
        ({"save_to_file": False}, False),
        ({"country_iso_code": "USA"}, True),
    ],
)
def test_codecarbon_offline_attribute_always_exists(config, expected_offline):
    """Test that codecarbon_offline attribute always exists regardless of config."""
    emissions = Emissions(codecarbon_config=config)
    assert hasattr(emissions, "codecarbon_offline")
    assert emissions.codecarbon_offline == expected_offline
