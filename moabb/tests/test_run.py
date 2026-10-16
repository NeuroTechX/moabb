import pytest

import moabb.run
from moabb.run import parser_init


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["-e", "WithinSession"], ["WithinSession"]),
        (["-e", "WithinSession", "CrossSession"], ["WithinSession", "CrossSession"]),
        ([], None),
    ],
)
def test_parser_evaluations(argv, expected):
    assert parser_init().parse_args(argv).evaluations == expected


def test_main_forwards_options_to_benchmark(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(moabb.run, "benchmark", lambda **kw: calls.append(kw))
    monkeypatch.setattr(moabb.run, "set_download_dir", calls.append)
    argv = [
        "--pipelines",
        str(tmp_path),
        "-e",
        "CrossSession",
        "--results",
        str(tmp_path / "res"),
        "--mne_data",
        str(tmp_path / "data"),
        "--threads",
        "2",
        "-f",
    ]
    assert moabb.run.main(argv) == 0
    assert calls[0] == str(tmp_path / "data")
    assert calls[1] == {
        "pipelines": str(tmp_path),
        "evaluations": ["CrossSession"],
        "results": str(tmp_path / "res"),
        "overwrite": True,
        "output": "./",
        "n_jobs": 2,
        "plot": False,
        "contexts": None,
    }


def test_main_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        moabb.run.main(["--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("usage: moabb")
