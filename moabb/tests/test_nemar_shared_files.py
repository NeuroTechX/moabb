"""Per-subject NEMAR prefetch also fetches the files no subject owns (offline)."""

import glob
import json

import pytest

import moabb.datasets.download as dl


def _store(tmp_path, entries, present=()):
    target = tmp_path / "NEMAR" / "nm000329"
    provenance = target / dl.SOURCEDATA_PROVENANCE
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"dataset": "nm000329", "files": entries}))
    for name in present:
        (target / "sourcedata" / name).parent.mkdir(parents=True, exist_ok=True)
        (target / "sourcedata" / name).write_text("x")
    return target


# Shapes from nm000329 (Brandl2020): the montage and band/interval definitions
# carry "subject": null and every subject's load reads them.
_BRANDL = [
    {"file": "pp1.mat", "subject": "1"},
    {"file": "pp2.mat", "subject": "2"},
    {"file": "mnt.mat", "subject": None},
    {"file": "bands.mat", "subject": ""},
    {"file": "ivals.mat"},
]


def test_subject_selection_includes_shared_files(tmp_path):
    target = _store(tmp_path, _BRANDL)

    patterns = dl._sourcedata_files_for_subject(target, "nm000329", 1, force_update=False)

    assert patterns == [
        "sourcedata/" + glob.escape(name)
        for name in ("pp1.mat", "mnt.mat", "bands.mat", "ivals.mat")
    ]


def test_shared_files_do_not_make_an_unknown_subject_valid(tmp_path):
    target = _store(tmp_path, _BRANDL)

    with pytest.raises(dl.NemarDownloadError, match="lists no sourcedata for subject"):
        dl._sourcedata_files_for_subject(target, "nm000329", 99, force_update=False)


def test_a_store_missing_the_shared_files_is_not_local(tmp_path):
    target = _store(tmp_path, _BRANDL, present=("pp1.mat",))
    assert not dl.nemar_sourcedata_is_local(target, 1)

    for name in ("mnt.mat", "bands.mat", "ivals.mat"):
        (target / "sourcedata" / name).write_text("x")
    assert dl.nemar_sourcedata_is_local(target, 1)
    assert not dl.nemar_sourcedata_is_local(target, 2)


def test_manifests_without_subjects_keep_the_whole_tree_rule(tmp_path):
    entries = [{"file": "a.mat"}, {"file": "b.mat"}]
    target = _store(tmp_path, entries, present=("a.mat",))

    assert dl._manifest_shared_files({"files": entries}) == []
    assert dl._sourcedata_files_for_subject(target, "nm000329", 1, False) is None
    assert dl.nemar_sourcedata_is_local(target, 1)
