"""Shared utility functions for the moabb.analysis package."""

from __future__ import annotations

import re


_RAISE = object()  # sentinel distinct from None


def _match_int(s, default=_RAISE):
    """Match the first integer in a string.

    Parameters
    ----------
    s : str
        String to search for an integer.
    default : int, None, or _RAISE, optional
        Value to return if no integer is found.  The default sentinel
        ``_RAISE`` causes an ``AssertionError``; any other value
        (including ``None``) is returned as-is.

    Returns
    -------
    int or default
        The first integer found in the string, or *default* if not found.
    """
    match = re.search(r"(\d+)", str(s))
    if match is None:
        if default is not _RAISE:
            return default
        raise AssertionError(f"Cannot parse number from '{s}'")
    return int(match.group(1))


def _match_float(s):
    """Match the first float in a string."""
    match = re.search(r"(\d+\.?\d*)", str(s))
    assert match, f"Cannot parse float from '{s}'"
    return float(match.group(1))


def _compute_n_trials(row, paradigm):
    """Compute total number of trials per session from a summary table row.

    Parameters
    ----------
    row : dict
        A dataset's ``_summary_table`` dictionary.
    paradigm : str
        The dataset paradigm (e.g. ``"imagery"``, ``"p300"``).

    Returns
    -------
    int or None
        Total number of trials, or None if parsing fails.
    """
    if paradigm in ("imagery", "ssvep"):
        trials_per_class = _match_int(row.get("#Trials / class", ""), default=None)
        n_classes = _match_int(row.get("#Classes", ""), default=None)
        if trials_per_class is not None and n_classes is not None:
            return trials_per_class * n_classes
    elif paradigm == "rstate":
        n_classes = _match_int(row.get("#Classes", ""), default=None)
        n_blocks = _match_int(row.get("#Blocks / class", ""), default=None)
        if n_classes is not None and n_blocks is not None:
            return n_classes * n_blocks
    elif paradigm == "cvep":
        trials_per_class = _match_int(row.get("#Trials / class", ""), default=None)
        n_trial_classes = _match_int(row.get("#Trial classes", ""), default=None)
        if trials_per_class is not None and n_trial_classes is not None:
            return trials_per_class * n_trial_classes
    else:  # p300
        trial_str = row.get("#Trials / class", "")
        match = re.search(r"(\d+) NT / (\d+) T", str(trial_str))
        if match is not None:
            return int(match.group(1)) + int(match.group(2))
        return _match_int(trial_str, default=None)
    return None
