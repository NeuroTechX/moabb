"""Minimal XDF reader.

Covers what MOABB's XDF-based datasets need: regular numeric streams and
irregular string-marker streams, returned in ``pyxdf``'s result shape so
loaders can switch between the two without logic changes. Regular-rate
streams get ``pyxdf``'s dejittering (per-segment linear fit of timestamps),
so sample-index lookups like ``np.searchsorted`` place events exactly where
``pyxdf`` would; no clock synchronization is applied, which is adequate when
all streams were recorded on one machine (a single shared clock), true for
every dataset this backs. The stream footer is ignored
entirely, so recorders that write non-conforming footers (e.g.
AguileraRodriguez2025's float ``sample_count``, which makes ``pyxdf``
1.17.5 raise ``ValueError``) load fine.
"""

import struct
from xml.etree import ElementTree

import numpy as np


_FMT = {
    "float32": ("<f4", 4),
    "double64": ("<f8", 8),
    "int8": ("<i1", 1),
    "int16": ("<i2", 2),
    "int32": ("<i4", 4),
    "int64": ("<i8", 8),
}


def _varlen(f):
    n = f.read(1)[0]
    return int.from_bytes(f.read(n), "little")


def _xml2dict(element):
    """Nest an XML element the way ``pyxdf`` does: children as lists."""
    out = {}
    for child in element:
        out.setdefault(child.tag, []).append(
            _xml2dict(child) if len(child) else child.text
        )
    return out


def _dejitter(stamps, srate, threshold_seconds=1.0, threshold_samples=500):
    """Replace timestamps with per-segment linear fits, as ``pyxdf`` does."""
    tdiff = 1.0 / srate
    breaks = np.abs(np.diff(stamps)) > max(threshold_seconds, threshold_samples * tdiff)
    bounds = [0, *np.flatnonzero(breaks) + 1, len(stamps)]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        idx = np.arange(start, stop, 1)[:, None]
        X = np.concatenate((np.ones_like(idx), idx), axis=1)
        mapping = np.linalg.lstsq(X, stamps[idx], rcond=-1)[0]
        stamps[idx] = mapping[0] + mapping[1] * idx
    return stamps


def read_xdf(path):
    """Read an XDF file.

    Returns
    -------
    streams : list of dict
        One dict per stream in file order, shaped like ``pyxdf.load_xdf``'s:
        ``info`` (XML fields as lists, including nested ``desc``),
        ``time_series`` (``ndarray (n_samples, n_channels)`` for numeric
        formats, list of per-sample string lists otherwise) and
        ``time_stamps`` (``ndarray`` of raw LSL seconds).
    header : dict
        The file header's ``info`` fields, same nesting.
    """
    streams, order, header = {}, [], {}
    with open(path, "rb") as f:
        if f.read(4) != b"XDF:":
            raise ValueError(f"{path} is not an XDF file")
        while True:
            head = f.read(1)
            if not head:
                break
            length = int.from_bytes(f.read(head[0]), "little")
            tag = struct.unpack("<H", f.read(2))[0]
            body_len = length - 2
            if tag == 1:  # FileHeader
                header = _xml2dict(ElementTree.fromstring(f.read(body_len).decode()))
            elif tag == 2:  # StreamHeader
                sid = struct.unpack("<I", f.read(4))[0]
                xml = ElementTree.fromstring(f.read(body_len - 4).decode())
                streams[sid] = {
                    "info": _xml2dict(xml),
                    "time_series": [],
                    "time_stamps": [],
                }
                order.append(sid)
            elif tag == 3:  # Samples
                sid = struct.unpack("<I", f.read(4))[0]
                s = streams[sid]
                info = s["info"]
                fmt = info["channel_format"][0]
                n_ch = int(info["channel_count"][0])
                srate = float(info["nominal_srate"][0] or 0)
                dt = 1.0 / srate if srate else 0.0
                last = s["time_stamps"][-1] if s["time_stamps"] else 0.0
                for _ in range(_varlen(f)):
                    stamp = (
                        struct.unpack("<d", f.read(8))[0] if f.read(1)[0] else last + dt
                    )
                    last = stamp
                    s["time_stamps"].append(stamp)
                    if fmt == "string":
                        s["time_series"].append(
                            [f.read(_varlen(f)).decode() for _ in range(n_ch)]
                        )
                    else:
                        s["time_series"].append(f.read(n_ch * _FMT[fmt][1]))
            else:  # ClockOffset, Boundary, StreamFooter -- not needed
                f.seek(body_len, 1)
    for s in streams.values():
        fmt = s["info"]["channel_format"][0]
        if fmt != "string":
            n_ch = int(s["info"]["channel_count"][0])
            s["time_series"] = np.frombuffer(
                b"".join(s["time_series"]), dtype=_FMT[fmt][0]
            ).reshape(-1, n_ch)
        s["time_stamps"] = np.asarray(s["time_stamps"])
        srate = float(s["info"]["nominal_srate"][0] or 0)
        if srate > 0 and len(s["time_stamps"]) > 1:
            s["time_stamps"] = _dejitter(s["time_stamps"], srate)
    return [streams[sid] for sid in order], header
