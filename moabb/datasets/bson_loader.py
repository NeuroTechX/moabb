"""Minimal BSON parser for reading MEDUSA recording files.

This module provides a lightweight BSON reader that doesn't require any
external dependencies. It supports the BSON types used in MEDUSA recording
files.

BSON (Binary JSON) is a binary-encoded serialization of JSON-like documents.
See https://bsonspec.org/ for the specification.
"""

import struct


def _read_cstring(data, offset):
    """Read a null-terminated C string from data."""
    end = data.index(b"\x00", offset)
    return data[offset:end].decode("utf-8"), end + 1


def _read_document(data, offset=0):
    """Read a BSON document from data at the given offset.

    Returns the parsed document as a dictionary and the new offset.
    """
    doc_size = struct.unpack_from("<i", data, offset)[0]
    end = offset + doc_size - 1  # -1 for trailing null byte
    offset += 4

    result = {}
    while offset < end:
        elem_type = data[offset]
        offset += 1

        # Read element name (cstring)
        name, offset = _read_cstring(data, offset)

        # Parse value based on type
        if elem_type == 0x01:  # 64-bit float (double)
            value = struct.unpack_from("<d", data, offset)[0]
            offset += 8

        elif elem_type == 0x02:  # UTF-8 string
            str_len = struct.unpack_from("<i", data, offset)[0]
            offset += 4
            value = data[offset : offset + str_len - 1].decode("utf-8")
            offset += str_len

        elif elem_type == 0x03:  # Embedded document
            value, offset = _read_document(data, offset)

        elif elem_type == 0x04:  # Array
            arr_doc, offset = _read_document(data, offset)
            # Convert dict with string indices "0", "1", ... to list
            value = [arr_doc[str(i)] for i in range(len(arr_doc))]

        elif elem_type == 0x05:  # Binary data
            bin_len = struct.unpack_from("<i", data, offset)[0]
            offset += 4
            _ = data[offset]  # BSON binary subtype (unused but must be read)
            offset += 1
            value = data[offset : offset + bin_len]
            offset += bin_len

        elif elem_type == 0x08:  # Boolean
            value = data[offset] != 0
            offset += 1

        elif elem_type == 0x09:  # UTC datetime (int64 milliseconds since epoch)
            ms = struct.unpack_from("<q", data, offset)[0]
            offset += 8
            value = ms  # Keep as milliseconds, let caller convert if needed

        elif elem_type == 0x0A:  # Null
            value = None

        elif elem_type == 0x10:  # 32-bit integer
            value = struct.unpack_from("<i", data, offset)[0]
            offset += 4

        elif elem_type == 0x12:  # 64-bit integer
            value = struct.unpack_from("<q", data, offset)[0]
            offset += 8

        else:
            raise ValueError(
                f"Unsupported BSON type: {hex(elem_type)} for field '{name}'"
            )

        result[name] = value

    return result, offset + 1  # +1 for trailing null


def load_bson(file_path):
    """Load a BSON file and return the parsed document as a dictionary.

    Parameters
    ----------
    file_path : str or Path
        Path to the BSON file.

    Returns
    -------
    dict
        The parsed BSON document.

    Examples
    --------
    >>> data = load_bson("recording.bson")
    >>> print(data.keys())
    dict_keys(['subject_id', 'recording_id', 'eeg', 'cvepspellerdata', ...])
    """
    with open(file_path, "rb") as f:
        data = f.read()
    result, _ = _read_document(data, 0)
    return result
