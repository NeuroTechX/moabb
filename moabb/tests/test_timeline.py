"""Tests for moabb.analysis.timeline module."""

import unittest
from unittest.mock import patch

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from moabb.datasets.fake import FakeDataset  # noqa: E402


class TestNormalizeClassLabel(unittest.TestCase):
    """Tests for _normalize_class_label helper."""

    def test_identity(self):
        from moabb.analysis.timeline import _normalize_class_label

        assert _normalize_class_label("target") == "target"

    def test_case_insensitive(self):
        from moabb.analysis.timeline import _normalize_class_label

        assert _normalize_class_label("Target") == "target"
        assert _normalize_class_label("TARGET") == "target"

    def test_strips_non_alnum(self):
        from moabb.analysis.timeline import _normalize_class_label

        assert _normalize_class_label("non-target") == "nontarget"
        assert _normalize_class_label("left_hand") == "lefthand"
        assert _normalize_class_label("  Right Hand  ") == "righthand"

    def test_empty_string(self):
        from moabb.analysis.timeline import _normalize_class_label

        assert _normalize_class_label("") == ""


class TestExtractStimulusTimeline(unittest.TestCase):
    """Tests for extract_stimulus_timeline with FakeDataset."""

    def test_returns_stimulus_timeline(self):
        from moabb.analysis.timeline import StimulusTimeline, extract_stimulus_timeline

        ds = FakeDataset(paradigm="imagery")
        result = extract_stimulus_timeline(ds)
        assert isinstance(result, StimulusTimeline)

    def test_imagery_paradigm(self):
        from moabb.analysis.timeline import extract_stimulus_timeline

        ds = FakeDataset(paradigm="imagery")
        result = extract_stimulus_timeline(ds)
        assert result.paradigm == "imagery"

    def test_p300_paradigm(self):
        from moabb.analysis.timeline import extract_stimulus_timeline

        ds = FakeDataset(paradigm="p300", event_list=("Target", "NonTarget"))
        result = extract_stimulus_timeline(ds)
        assert result.paradigm == "p300"


class TestPlotClassBalance(unittest.TestCase):
    """Tests for plot_class_balance."""

    def tearDown(self):
        plt.close("all")

    def test_basic_rendering(self):
        from moabb.analysis.timeline import plot_class_balance

        ds = FakeDataset(paradigm="imagery")
        fig = plot_class_balance(ds)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_no_event_id_returns_none(self):
        from moabb.analysis.timeline import plot_class_balance

        ds = FakeDataset(paradigm="imagery")
        ds.event_id = {}
        fig = plot_class_balance(ds)
        assert fig is None

    def test_zero_count_fallback(self):
        """When metadata class labels don't match event_id keys, the chart
        should fall back to the 'no counts' display instead of showing zeros."""
        from moabb.analysis.timeline import plot_class_balance

        ds = FakeDataset(paradigm="imagery")

        # Simulate metadata with trial counts that don't match event_id keys
        class FakeDataStructure:
            n_trials_per_class = {"ClassX": 50, "ClassY": 50}

        class FakeMetadata:
            data_structure = FakeDataStructure()

        with patch.object(type(ds), "METADATA", FakeMetadata(), create=True):
            fig = plot_class_balance(ds)
            assert fig is not None
            ax = fig.axes[0]
            # Should show "counts vary by subject" title (the fallback)
            # rather than "Balanced: 0 trials/class"
            title = ax.get_title()
            assert "0 trials/class" not in title

    def test_matched_labels(self):
        """When metadata keys match event_id, counts should be displayed."""
        from moabb.analysis.timeline import plot_class_balance

        ds = FakeDataset(paradigm="imagery", event_list=("left_hand", "right_hand"))

        class FakeDataStructure:
            n_trials_per_class = {"left_hand": 40, "right_hand": 60}

        class FakeMetadata:
            data_structure = FakeDataStructure()

        with patch.object(type(ds), "METADATA", FakeMetadata(), create=True):
            fig = plot_class_balance(ds)
            assert fig is not None
            ax = fig.axes[0]
            title = ax.get_title()
            assert "Trial counts per class" in title


class TestPlotSessionStructure(unittest.TestCase):
    """Tests for plot_session_structure."""

    def tearDown(self):
        plt.close("all")

    def test_basic_rendering(self):
        from moabb.analysis.timeline import plot_session_structure

        ds = FakeDataset(paradigm="imagery", n_sessions=2)
        fig = plot_session_structure(ds)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_no_sessions_returns_none(self):
        from moabb.analysis.timeline import plot_session_structure

        ds = FakeDataset(paradigm="imagery")
        ds.n_sessions = None
        fig = plot_session_structure(ds)
        assert fig is None


if __name__ == "__main__":
    unittest.main()
