"""Minimal XDF reader.

Covers what MOABB's XDF-based datasets need: regular numeric streams and
irregular string-marker streams, returned in ``pyxdf``'s result shape so
loaders can switch between the two without logic changes. Regular-rate
streams get ``pyxdf``'s dejittering (per-segment linear fit of timestamps),
so sample-index lookups like ``np.searchsorted`` place events exactly where
``pyxdf`` would; clock
synchronization is applied exactly as ``pyxdf`` does (Huber-ADMM robust fit
of the recorded clock offsets, ported from pyxdf, BSD-2-Clause), so results
are interchangeable with pyxdf's. The stream footer is ignored
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


def _robust_fit(A, y, rho=1, iters=1000):
    """Huber-loss linear regression via ADMM. Ported from pyxdf (BSD-2)."""
    A = np.copy(A)
    offset = np.min(A[:, 1])
    A[:, 1] -= offset
    Aty = np.dot(A.T, y)
    L = np.linalg.cholesky(np.dot(A.T, A))
    U = L.T
    z = np.zeros_like(y)
    u = z
    x = z
    for _ in range(iters):
        x = np.linalg.solve(U, (np.linalg.solve(L, Aty + np.dot(A.T, z - u))))
        d = np.dot(A, x) - y + u
        d_inv = np.zeros_like(d)
        np.divide(1, d, out=d_inv, where=d != 0)
        tmp = np.maximum(0, (1 - (1 + 1 / rho) * np.abs(d_inv)))
        z = rho / (1 + rho) * d + 1 / (1 + rho) * tmp * d
        u = d - z
    x[0] -= x[1] * offset
    return x


def _segment_clock_diff(diff, thresh_stds, thresh_secs):
    """MAD-standardised glitch detection. Ported from pyxdf (BSD-2)."""
    median = np.median(diff)
    diffs_shift = diff - median
    diffs_shift_abs = np.abs(diffs_shift)
    mad = np.median(diffs_shift_abs) + np.finfo(float).eps
    diffs_std = diffs_shift / mad
    return (np.abs(diffs_std) > thresh_stds) & (diffs_shift_abs > thresh_secs)


def _clock_reset_ranges(clock_times, clock_values):
    """Inclusive index ranges between clock resets. Ported from pyxdf (BSD-2)."""
    time_diff = np.diff(clock_times)
    value_diff = np.diff(clock_values)
    resets_at = (time_diff < 0) | (
        _segment_clock_diff(time_diff, 5, 5) & _segment_clock_diff(value_diff, 10, 1)
    )
    break_inds = np.where(resets_at)[0]
    starts = np.hstack(([0], break_inds + 1))
    ends = np.hstack((break_inds, len(resets_at)))
    return list(zip(starts.tolist(), ends.tolist()))


def _clock_sync(stamps, clock_times, clock_values, winsor_threshold=0.0001):
    """Apply pyxdf's clock-offset correction to one stream's timestamps."""
    if not len(stamps) or not clock_times:
        return stamps
    if len(clock_times) > 1:
        ranges = _clock_reset_ranges(clock_times, clock_values)
    else:
        ranges = [(0, 0)]
    coef = []
    for start, stop_inclusive in ranges:
        if start != stop_inclusive:
            stop = stop_inclusive + 1
            X = np.column_stack(
                [
                    np.ones(stop - start),
                    np.array(clock_times[start:stop]) / winsor_threshold,
                ]
            )
            y = np.array(clock_values[start:stop]) / winsor_threshold
            try:
                coefs = _robust_fit(X, y)
                coefs[0] *= winsor_threshold
            except np.linalg.LinAlgError:
                coefs = [0, 0]
            coef.append(coefs)
        else:
            coef.append((clock_values[start], 0))
    if len(ranges) == 1:
        stamps += coef[0][0] + (coef[0][1] * stamps)
        return stamps
    ts_start = 0
    for coef_i, range_i in zip(coef, ranges):
        stop = range_i[1] + 1
        if stop < len(clock_times):
            current_end_t = clock_times[range_i[1]]
            next_start_t = clock_times[stop]
            cond = np.less(
                np.abs(stamps[ts_start:] - current_end_t),
                np.abs(stamps[ts_start:] - next_start_t),
            )
            ts_stop = ts_start + (len(cond) if all(cond) else np.argmin(cond).item())
        else:
            ts_stop = len(stamps)
        if ts_start != ts_stop:
            ts_slice = slice(ts_start, ts_stop)
            stamps[ts_slice] += coef_i[0] + coef_i[1] * stamps[ts_slice]
        ts_start = ts_stop
    return stamps


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
                    "clock_times": [],
                    "clock_values": [],
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
            elif tag == 4:  # ClockOffset
                sid = struct.unpack("<I", f.read(4))[0]
                t, off = struct.unpack("<dd", f.read(16))
                streams[sid]["clock_times"].append(t)
                streams[sid]["clock_values"].append(off)
            else:  # Boundary, StreamFooter -- not needed
                f.seek(body_len, 1)
    for s in streams.values():
        fmt = s["info"]["channel_format"][0]
        if fmt != "string":
            n_ch = int(s["info"]["channel_count"][0])
            s["time_series"] = np.frombuffer(
                b"".join(s["time_series"]), dtype=_FMT[fmt][0]
            ).reshape(-1, n_ch)
        s["time_stamps"] = _clock_sync(
            np.asarray(s["time_stamps"]), s.pop("clock_times"), s.pop("clock_values")
        )
        srate = float(s["info"]["nominal_srate"][0] or 0)
        if srate > 0 and len(s["time_stamps"]) > 1:
            s["time_stamps"] = _dejitter(s["time_stamps"], srate)
    return [streams[sid] for sid in order], header
