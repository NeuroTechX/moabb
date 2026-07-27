"""Regression tests for Iwama2023 downloads."""


import requests

from moabb.datasets.iwama2023 import Iwama2023


class _Response:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def test_download_retries_stream_failure_and_keeps_only_complete_file(
    tmp_path, monkeypatch
):
    """A transient S3 stream failure retries without exposing a partial file."""
    first = _Response([b"partial", requests.ConnectionError("interrupted")])
    second = _Response([b"complete"])
    responses = iter([first, second])

    monkeypatch.setattr(
        "moabb.datasets.iwama2023.requests.get", lambda *args, **kwargs: next(responses)
    )

    assert Iwama2023._download_file(tmp_path, "sub-020/example.edf", False)

    target = tmp_path / "sub-020" / "example.edf"
    assert target.read_bytes() == b"complete"
    assert first.closed and second.closed
    assert not list(target.parent.glob("tmp*"))


def test_download_does_not_replace_existing_file_without_force_update(
    tmp_path, monkeypatch
):
    """An already complete local file avoids an unnecessary network request."""
    target = tmp_path / "example.edf"
    target.write_bytes(b"cached")
    monkeypatch.setattr(
        "moabb.datasets.iwama2023.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected request")
        ),
    )

    assert Iwama2023._download_file(tmp_path, "example.edf", False)
    assert target.read_bytes() == b"cached"
