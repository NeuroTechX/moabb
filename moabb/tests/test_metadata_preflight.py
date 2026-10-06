"""Keep the documented paradigm-only metadata recipe free of runtime work."""

import builtins
import io
import json
import os
import socket
from copy import deepcopy
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

import moabb
from moabb.analysis import Results
from moabb.datasets import BNCI2014_001, BNCI2014_009, AlexMI
from moabb.datasets.base import BaseDataset
from moabb.evaluations.base import BaseEvaluation
from moabb.paradigms import MotorImagery
from moabb.paradigms.base import BaseParadigm


_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/data_management_and_configuration/plot_metadata_preflight.py"
)


def _forbid(*args, **kwargs):
    raise AssertionError("metadata preflight must not perform I/O or runtime work")


def test_documented_metadata_preflight(monkeypatch):
    # Load the source before forbidding I/O; imports have already been performed.
    source = compile(_EXAMPLE.read_text(), str(_EXAMPLE), "exec")
    dataset_types = (AlexMI, BNCI2014_001, BNCI2014_009)
    datasets = [dataset_type() for dataset_type in dataset_types]
    strict = MotorImagery(events=["left_hand", "right_hand"], n_classes=2)
    default = MotorImagery(events=["left_hand", "right_hand"])
    datasets_before = deepcopy([vars(dataset) for dataset in datasets])
    paradigms_before = deepcopy([vars(strict), vars(default)])
    environment_before = dict(os.environ)
    output = []

    with monkeypatch.context() as patch:
        for module, name in (
            (builtins, "open"),
            (io, "open"),
            (os, "open"),
            (os, "mkdir"),
            (socket, "socket"),
            (socket, "create_connection"),
            (socket, "getaddrinfo"),
            (Results, "__init__"),
            (BaseEvaluation, "__init__"),
            (BaseDataset, "get_data"),
            (BaseDataset, "download"),
            (BaseDataset, "data_path"),
            (BaseParadigm, "get_data"),
            (MotorImagery, "used_events"),
            (Pipeline, "fit"),
        ):
            patch.setattr(module, name, _forbid)
        for dataset_type in dataset_types:
            patch.setattr(dataset_type, "data_path", _forbid)
            patch.setattr(dataset_type, "_get_single_subject_data", _forbid)
        patch.setattr(MotorImagery, "datasets", property(_forbid))
        patch.setattr(builtins, "print", lambda value: output.append(value))

        namespace = {}
        exec(source, namespace)
        # Existing native predicates are the sole rule source. Check their
        # default/explicit class behavior and the non-imagery rejection.
        assert [strict.is_valid(dataset) for dataset in datasets] == [False, True, False]
        assert [default.is_valid(dataset) for dataset in datasets] == [True, True, False]
        with pytest.raises(AttributeError):
            strict.is_valid({"n_sessions": None})
        assert [vars(dataset) for dataset in datasets] == datasets_before
        assert [vars(strict), vars(default)] == paradigms_before
        assert dict(os.environ) == environment_before

    report = json.loads(output[0])
    assert report == [
        {
            "dataset": dataset.code,
            "moabb_version": moabb.__version__,
            "paradigm_compatible": compatible,
            "declared_sessions": sessions,
            "evaluation_compatible": None,
        }
        for dataset, compatible, sessions in zip(datasets[:2], (False, True), (1, 2))
    ]
    assert output[1:3] == [True, False]
    assert json.loads(output[3]) == {
        "id": "synthetic-unresolved-record",
        "n_sessions": None,
    }
    assert namespace["paradigm"].n_classes == 2
    assert namespace["unspecified_classes"].n_classes is None
    assert len(namespace["datasets"]) == 2
