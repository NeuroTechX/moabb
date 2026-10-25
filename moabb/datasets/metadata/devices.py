"""Canonical recording-device names, for grouping trials by device.

``METADATA.acquisition.hardware`` is free text: one amplifier is written many
ways ("g.USBamp (g.tec)", "g.tec g.USBamp", "g.USBAmp (g.tec, Austria)").
:func:`canonical_device` maps it to one name per amplifier family, the value
of the ``device`` column of the paradigm metadata. A dataset naming only the
manufacturer gets ``"<manufacturer> (model not given)"``, so it is not
silently grouped with a specific model; an unmatched string is kept as is.
"""

import re


# Ordered: specific models before the manufacturer fallbacks.
# ponytail: a pattern table, not a device registry; add a row per new family.
_DEVICES = (
    (r"usb ?amp", "g.tec g.USBamp"),
    (r"nautilus", "g.tec g.Nautilus"),
    (r"hiamp", "g.tec g.HIamp"),
    (r"mobilab", "g.tec g.MOBIlab"),
    (r"unicorn", "g.tec Unicorn"),
    (r"g\.?tec", "g.tec (model not given)"),
    (r"brain ?amp", "Brain Products BrainAmp"),
    (r"actichamp", "Brain Products actiCHamp"),
    (r"liveamp", "Brain Products LiveAmp"),
    (r"brain ?products|brain ?vision", "Brain Products (model not given)"),
    (r"biosemi", "BioSemi ActiveTwo"),
    (r"synamps", "Neuroscan SynAmps"),
    (r"\begi\b|geodesic", "EGI Geodesic Net Amps"),
    (r"emotiv", "Emotiv EPOC"),
    (r"eego", "ANT Neuro eego"),
    (r"ant neuro", "ANT Neuro (model not given)"),
    (r"neuracle|neusen", "Neuracle NeuSen"),
    (r"smarting|mbraintrain", "mBrainTrain Smarting"),
    (r"nexus", "MindMedia NeXus"),
    (r"openbci|cyton", "OpenBCI Cyton"),
    (r"zhentec", "ZhenTec"),
    (r"dsi-", "Wearable Sensing DSI"),
)


def canonical_device(hardware):
    """Canonical device name of a free-text ``hardware`` description.

    Parameters
    ----------
    hardware : str | None
        ``METADATA.acquisition.hardware`` of a dataset.

    Returns
    -------
    str
        The amplifier family (e.g. ``"g.tec g.USBamp"``), the text itself when
        no family matches, or ``"unknown"`` when ``hardware`` is empty.
    """
    if not hardware or not str(hardware).strip():
        return "unknown"
    text = str(hardware).lower()
    for pattern, name in _DEVICES:
        if re.search(pattern, text):
            return name
    return str(hardware).strip()
