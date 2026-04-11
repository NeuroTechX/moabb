"""Shared constants and helpers for dataset documentation extensions.

Used by both ``dataset_timeline_ext`` and ``macro_table_ext`` (via
``scripts/generate_macro_table.py``).
"""

from __future__ import annotations

import pycountry


# ---------------------------------------------------------------------------
# Paradigm display metadata
# ---------------------------------------------------------------------------

PARADIGM_LABELS = {
    "imagery": "Imagery",
    "p300": "P300 / ERP",
    "erp": "P300 / ERP",
    "ssvep": "SSVEP",
    "cvep": "c-VEP",
    "rstate": "Resting State",
}

PARADIGM_COLORS = {
    "imagery": "#1565C0",
    "p300": "#D32F2F",
    "erp": "#D32F2F",
    "ssvep": "#2E7D32",
    "cvep": "#00695C",
    "rstate": "#546E7A",
}


# ---------------------------------------------------------------------------
# Country helpers
# ---------------------------------------------------------------------------


def normalize_country(raw: str | None) -> str | None:
    """Normalize a country string to ISO 3166-1 alpha-2 code using pycountry.

    Parameters
    ----------
    raw : str or None
        Country name, ISO alpha-2 code, or ISO alpha-3 code.

    Returns
    -------
    str or None
        Two-letter ISO 3166-1 alpha-2 code, or None if not recognized.
    """
    if not raw:
        return None
    raw = raw.strip()
    if len(raw) == 2:
        return raw.upper()
    try:
        return pycountry.countries.lookup(raw).alpha_2
    except LookupError:
        return None


def country_flag(code: str | None) -> str:
    """Return a flag emoji for an ISO 3166-1 alpha-2 country code.

    Parameters
    ----------
    code : str or None
        Two-letter country code (e.g. ``"FR"``, ``"US"``).

    Returns
    -------
    str
        Flag emoji string, or empty string if code is invalid.
    """
    if not code or len(code) != 2:
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code.upper())


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------


def normalize_health(status: str) -> str:
    """Normalize health status to ``'healthy'``, ``'patients'``, or ``'mixed'``.

    Non-standard values (e.g. ``"ALS patients"``, ``"stroke (recovery phase)"``)
    are mapped to ``"patients"``.  The original value can be preserved separately
    for tooltip display.

    Parameters
    ----------
    status : str
        Raw health status string from dataset metadata.

    Returns
    -------
    str
        Normalized category: ``"healthy"``, ``"patients"``, ``"mixed"``, or
        the original string if it cannot be classified.
    """
    if not status:
        return ""
    s = status.lower().strip()
    if s == "healthy":
        return "healthy"
    if "mixed" in s:
        return "mixed"
    if "patient" in s or "stroke" in s or "damage" in s or "pain" in s:
        return "patients"
    return status
