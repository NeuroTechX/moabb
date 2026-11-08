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

    def test_bnci2014_001_stimulus_protocol(self):
        from moabb.analysis.timeline import extract_stimulus_timeline
        from moabb.datasets import BNCI2014_001

        result = extract_stimulus_timeline(BNCI2014_001())

        assert result.total_duration_s == 6.0
        assert [(p.label, p.onset_s, p.duration_s) for p in result.phases] == [
            ("Fixation +", 0.0, 2.0),
            ("Cue + MI", 2.0, 1.25),
            ("Motor Imagery", 3.25, 2.75),
        ]

    def test_thapa2025_motor_execution_stimulus_protocol(self):
        from moabb.analysis.timeline import extract_stimulus_timeline
        from moabb.datasets import Thapa2025

        result = extract_stimulus_timeline(Thapa2025())

        assert result.total_duration_s == 19.0
        labels = [p.label for p in result.phases]
        assert "Fixation +" not in labels
        assert "Motor Imagery" not in labels
        assert [(p.label, p.onset_s, p.duration_s) for p in result.phases] == [
            ("Audio 'Start'", 0.0, 1.0),
            ("Self-Paced Motor Execution", 1.0, 11.0),
            ("Break / Rest", 12.0, 7.0),
        ]

    def test_self_paced_and_auditory_datasets_never_have_fixation_cross(self):
        from moabb.analysis.timeline import extract_stimulus_timeline
        from moabb.datasets import (
            BNCI2020_001,
            BNCI2025_001,
            BNCI2025_002,
            Brandl2020,
            Forenzo2023,
            Kueper2024,
            Rozado2015,
            Thapa2025,
            Wairagkar2018,
        )

        for cls in (
            Thapa2025,
            BNCI2020_001,
            Kueper2024,
            BNCI2025_001,
            BNCI2025_002,
            Wairagkar2018,
            Forenzo2023,
            Brandl2020,
            Rozado2015,
        ):
            tl = extract_stimulus_timeline(cls())
            labels = [p.label for p in tl.phases]
            assert "Fixation +" not in labels, f"{cls.__name__} had Fixation +: {labels}"

    def test_cued_and_cognitive_protocols_use_metadata_timings(self):
        from moabb.analysis.timeline import extract_stimulus_timeline
        from moabb.datasets import BNCI2024_001, HefmiIch2025, Weibo2014

        weibo = extract_stimulus_timeline(Weibo2014())
        assert [(p.label, p.onset_s, p.duration_s) for p in weibo.phases] == [
            ("Fixation +", 0.0, 2.0),
            ("Cue", 2.0, 1.0),
            ("Motor Imagery", 3.0, 4.0),
            ("Rest", 7.0, 1.0),
        ]

        hefmi = extract_stimulus_timeline(HefmiIch2025())
        assert [(p.label, p.onset_s, p.duration_s) for p in hefmi.phases] == [
            ("Cue", 0.0, 2.0),
            ("Motor Imagery", 2.0, 10.0),
            ("Rest", 12.0, 15.0),
        ]

        bnci24 = extract_stimulus_timeline(BNCI2024_001())
        assert bnci24.paradigm == "cognitive"
        assert [p.label for p in bnci24.phases] == [
            "Fade-In",
            "Letter Cue",
            "Fade-Out",
            "Handwriting",
        ]

    def test_non_motor_and_execution_task_labels(self):
        from moabb.analysis.timeline import extract_stimulus_timeline
        from moabb.datasets import (
            BNCI2025_001,
            BNCI2025_002,
            AguileraRodriguez2025,
            CastillosCVEP100,
            Gao2026,
            Gueho2026,
            Kaneshiro2015,
            Wairagkar2018,
        )

        kaneshiro = extract_stimulus_timeline(Kaneshiro2015())
        assert [p.label for p in kaneshiro.phases] == ["Visual Stimulus"]

        gao = extract_stimulus_timeline(Gao2026())
        assert any("Visual Imagery" in p.label for p in gao.phases)

        aguilera = extract_stimulus_timeline(AguileraRodriguez2025())
        assert any("Imagined Speech" in p.label for p in aguilera.phases)

        wairagkar = extract_stimulus_timeline(Wairagkar2018())
        assert any("Motor Execution" in p.label for p in wairagkar.phases)
        assert not any("Motor Imagery" in p.label for p in wairagkar.phases)

        bnci25_1 = extract_stimulus_timeline(BNCI2025_001())
        assert any("Motor Execution" in p.label for p in bnci25_1.phases)

        bnci25_2 = extract_stimulus_timeline(BNCI2025_002())
        assert any("Attempted Movement" in p.label for p in bnci25_2.phases)

        gueho = extract_stimulus_timeline(Gueho2026())
        assert any("Auditory Stimulus" in p.label for p in gueho.phases)

        castillos = extract_stimulus_timeline(CastillosCVEP100())
        assert "Feedback" not in [p.label for p in castillos.phases]


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
