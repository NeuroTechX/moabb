"""Verify that the maintainer-facing CV diagram is production-exact.

Never manually invent figure train/test/purge cells: every SVG matrix cell
must match the real PurgedEpochKFold splitter on the described toy events.
"""
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from moabb.evaluations.splitters import PurgedEpochKFold


def test_purged_epoch_kfold_svg_matches_actual_split_for_every_cell():
    n_epochs = 30
    events = np.arange(n_epochs) * 100
    epoch_length = 500
    labels = np.tile([0, 1], n_epochs // 2)
    X = np.zeros((n_epochs, 1))
    groups = np.empty((n_epochs, 3), dtype=object)
    groups[:, 0] = "one-continuous-run"
    groups[:, 1] = events
    groups[:, 2] = epoch_length

    splitter = PurgedEpochKFold(n_splits=5)
    folds = list(splitter.split(X, labels, groups))
    assert len(folds) == 5

    svg = Path(__file__).resolve().parents[2] / "docs/source/images/purged_epoch_kfold.svg"
    root = ElementTree.parse(svg).getroot()
    cells = {
        (int(el.attrib["data-fold"]), int(el.attrib["data-epoch"])): el.attrib["data-role"]
        for el in root.iter()
        if "data-role" in el.attrib
    }
    assert len(cells) == 5 * n_epochs

    all_epochs = set(range(n_epochs))
    for fold, (train, test) in enumerate(folds):
        training = set(train.tolist())
        testing = set(test.tolist())
        purged = all_epochs - training - testing

        # Dynamic-programming boundaries are chronological, not shuffled.
        assert len(testing) == 6
        assert max(testing) - min(testing) == 5
        assert len(purged) == (4 if fold in (0, 4) else 8)

        for i in range(n_epochs):
            expected = (
                "test" if i in testing
                else "train" if i in training
                else "purged"
            )
            assert cells[(fold, i)] == expected

        # Strict half-open interval purging: zero shared source samples.
        for i in training:
            assert all(
                not (events[i] < events[j] + epoch_length
                     and events[j] < events[i] + epoch_length)
                for j in testing
            )

    # Boundary witness drawn in the lower panel:
    # epoch 5 [500, 1000) and 6 [600, 1100) overlap; epoch 10
    # [1000, 1500) merely touches epoch 5 and is valid train.
    tr0, te0 = folds[0]
    assert 5 in te0
    assert 6 not in tr0 and 6 not in te0
    assert 10 in tr0
