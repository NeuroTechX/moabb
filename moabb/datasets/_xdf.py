"""Minimal XDF reader.

Covers what MOABB's XDF-based datasets need: regular numeric streams and
irregular string-marker streams, returned with their raw LSL timestamps.
No clock synchronization or dejittering is applied, which is adequate when
all streams were recorded on one machine (a single shared clock) -- true for
every dataset this backs. Unlike ``pyxdf``, the stream footer is ignored
entirely, so recorders that write non-conforming footers (e.g.
AguileraRodriguez2025's float ``sample_count``, which makes ``pyxdf`` 1.17.5
raise ``ValueError``) load fine.
"""

import struct
from xml.etree import ElementTree

import numpy as np


_FMT = {
    "float32": ("<f4", 4),
    "double64": ("<f8", 8),
    "int32": ("<i4", 4),
    "int16": ("<i2", 2),
    "int8": ("<i1", 1),
    "int64": ("<i8", 8),
}


def _varlen(f):
    n = f.read(1)[0]
    return int.from_bytes(f.read(n), "little")


def read_xdf(path):
    """Return {stream_id: {"info": dict, "series": ..., "stamps": np.ndarray,
    "clock_offsets": [(time, offset), ...]}} keyed by XDF stream id."""
    streams = {}
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
            if tag == 2:  # StreamHeader
                sid = struct.unpack("<I", f.read(4))[0]
                xml = ElementTree.fromstring(f.read(body_len - 4).decode())
                info = {c.tag: c.text for c in xml}
                streams[sid] = {
                    "info": info,
                    "series": [],
                    "stamps": [],
                    "clock_offsets": [],
                }
            elif tag == 3:  # Samples
                sid = struct.unpack("<I", f.read(4))[0]
                s = streams[sid]
                fmt = s["info"]["channel_format"]
                n_ch = int(s["info"]["channel_count"])
                n = _varlen(f)
                last = s["stamps"][-1] if s["stamps"] else 0.0
                srate = float(s["info"].get("nominal_srate") or 0)
                dt = 1.0 / srate if srate else 0.0
                for _ in range(n):
                    ts_bytes = f.read(1)[0]
                    stamp = struct.unpack("<d", f.read(8))[0] if ts_bytes else last + dt
                    last = stamp
                    s["stamps"].append(stamp)
                    if fmt == "string":
                        s["series"].append(
                            [f.read(_varlen(f)).decode() for _ in range(n_ch)]
                        )
                    else:
                        np_fmt, size = _FMT[fmt]
                        s["series"].append(f.read(n_ch * size))
            elif tag == 4:  # ClockOffset
                sid = struct.unpack("<I", f.read(4))[0]
                t, off = struct.unpack("<dd", f.read(16))
                streams[sid]["clock_offsets"].append((t, off))
            else:  # FileHeader, Boundary, StreamFooter -- skip
                f.seek(body_len, 1)
    for s in streams.values():
        if s["info"]["channel_format"] != "string":
            np_fmt, _ = _FMT[s["info"]["channel_format"]]
            n_ch = int(s["info"]["channel_count"])
            s["series"] = np.frombuffer(b"".join(s["series"]), dtype=np_fmt).reshape(
                -1, n_ch
            )
        s["stamps"] = np.asarray(s["stamps"])
    return streams
