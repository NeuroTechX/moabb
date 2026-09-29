"""Standard-montage names across MNE versions.

Leaf module (imports only MNE) so that low-level modules such as
:mod:`moabb.datasets.bids_interface` can use it without an import cycle; it is
re-exported by :mod:`moabb.datasets.utils`.
"""

import mne


# MNE 1.13 renamed the template montages fitted on the Colin27 head and
# deprecated the old spellings; MNE 1.14 removes them.
_RENAMED_STANDARD_MONTAGES = {
    "standard_1005": "colin27_1005",
    "standard_1020": "colin27_1020",
    "standard_alphabetic": "colin27_alphabetic",
    "standard_postfixed": "colin27_postfixed",
    "standard_prefixed": "colin27_prefixed",
    "standard_primed": "colin27_primed",
}
_LEGACY_STANDARD_MONTAGES = {v: k for k, v in _RENAMED_STANDARD_MONTAGES.items()}


def resolve_montage_name(name):
    """Return the spelling of a standard montage known to the installed MNE.

    MNE 1.13 renamed ``standard_1005``/``standard_1020`` (and the other
    ``standard_*`` templates) to ``colin27_*``: the electrode files are
    byte-identical, the old names emit a ``FutureWarning`` and MNE 1.14
    removes them. MOABB supports older MNE releases that only know the old
    names, so loaders spell the new name and call this helper.

    Parameters
    ----------
    name : str
        A montage name in either spelling, e.g. ``"colin27_1005"``. Names
        that were not renamed (``"biosemi64"``, ``"GSN-HydroCel-129"``...)
        are returned unchanged.

    Returns
    -------
    str
        The new ``colin27_*`` name when the installed MNE provides it,
        otherwise the legacy ``standard_*`` name.
    """
    new = _RENAMED_STANDARD_MONTAGES.get(name, name)
    old = _LEGACY_STANDARD_MONTAGES.get(new)
    if old is None:
        return name
    return new if new in mne.channels.get_builtin_montages() else old
