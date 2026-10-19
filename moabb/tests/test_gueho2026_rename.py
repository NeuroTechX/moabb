"""Gueho2026 rename (was Lenaig2026) + W42 metadata fixes."""

import logging

import pytest

from moabb.datasets import Gueho2026, Lenaig2026
from moabb.datasets.base import BaseDataset


def test_gueho_is_real_class_with_unchanged_code():
    ds = Gueho2026()
    assert ds.code == "Lenaig2026"
    assert type(ds).__name__ == "Gueho2026"


def test_lenaig_alias_warns_and_works(caplog):
    with caplog.at_level(logging.WARNING):
        ds = Lenaig2026()
    assert isinstance(ds, Gueho2026)
    assert "renamed to Gueho2026" in caplog.text


def test_no_is_abbrev_warning_for_renamed_class(caplog):
    with caplog.at_level(logging.WARNING):
        Gueho2026()
    assert "must be an abbreviation" not in caplog.text


def test_readme_uses_capital_trials_heading():
    from moabb.datasets.bids_interface import _build_readme

    assert "Trials per class" in _build_readme(Gueho2026(exp=1, run="both"))


def test_metadata_fixes():
    from moabb.datasets import (
        BNCI2025_002,
        Daly2020,
        ErpCore2021_ERN,
        Han2026,
        Hinss2021,
        Lee2024_TV,
        Ofner2017,
        PerezBlanco2026,
    )

    assert "An EEG-EMG-kinematics dataset from wrist pointing tasks" in (
        PerezBlanco2026.__doc__ or ""
    )
    assert "Kobler" in BNCI2025_002.METADATA.documentation.investigators[0]
    assert Daly2020.METADATA.documentation.publication_year == 2020
    assert "5055046" in (Hinss2021.METADATA.documentation.data_url or "")
    assert "Basketball Motor Imagery EEG Dataset" in (Han2026.__doc__ or "")
    assert "Observation and Motor Imagery" not in (Han2026.__doc__ or "")
    assert ErpCore2021_ERN.METADATA.documentation.doi == (
        "10.1016/j.neuroimage.2020.117465"
    )
    assert ErpCore2021_ERN.METADATA.documentation.publication_year == 2021
    assert Lee2024_TV().METADATA.documentation.doi
    assert Lee2024_TV().METADATA.documentation.publication_year == 2024
    assert "Imagery ataset" not in (Ofner2017.__doc__ or "")
