"""Unified 4-stage smoke tests for all SSVEP dataset adapters.

Run with: SSVEP_TEST_DOWNLOAD=1 pytest ssvep_moabb_adapters/tests/test_smoke.py -v

Stage 1: data_path - Download/path resolution
Stage 2: _get_single_subject_data - Raw loading
Stage 3: paradigm.get_data - SSVEP epoch extraction
Stage 4: WithinSessionEvaluation - End-to-end benchmark
"""

import pytest
from ssvep_moabb_adapters.tests.conftest import requires_download


# All datasets to test with their configurations
DATASETS = [
    ("Liu2020BETA", "ssvep_moabb_adapters.liu2020_beta", "Liu2020BETA"),
    ("Liu2022EldBETA", "ssvep_moabb_adapters.chen2024_eldbeta", "Liu2022EldBETA"),
    ("Kim2025BetaRange", "ssvep_moabb_adapters.kim2025_beta_range", "Kim2025BetaRange"),
    ("Dong2023", "ssvep_moabb_adapters.dong2023_ssvep", "Dong2023"),
    ("Han2024Fatigue", "ssvep_moabb_adapters.han2024_fatigue", "Han2024Fatigue"),
    (
        "Chen2017SingleFlicker",
        "ssvep_moabb_adapters.chen2017_single_flicker",
        "Chen2017SingleFlicker",
    ),
    ("Wang2021Combined", "ssvep_moabb_adapters.wang2021_combined", "Wang2021Combined"),
    ("Lee2021Mobile_SSVEP", "ssvep_moabb_adapters.lee2021_mobile", "Lee2021Mobile_SSVEP"),
]


def _get_dataset(module_name, class_name):
    """Dynamically import and instantiate a dataset."""
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


@requires_download()
@pytest.mark.download
@pytest.mark.parametrize("name,module,cls_name", DATASETS, ids=[d[0] for d in DATASETS])
class TestStage1DataPath:
    """Stage 1: Verify data_path returns valid path(s)."""

    def test_data_path_subject_1(self, name, module, cls_name):
        dataset = _get_dataset(module, cls_name)
        subject = dataset.subject_list[0]
        result = dataset.data_path(subject)
        assert result is not None
        if isinstance(result, dict):
            assert len(result) > 0
            for v in result.values():
                assert v is not None
        elif isinstance(result, list):
            assert len(result) > 0
            for p in result:
                assert p is not None


@requires_download()
@pytest.mark.download
@pytest.mark.parametrize("name,module,cls_name", DATASETS, ids=[d[0] for d in DATASETS])
class TestStage2RawLoading:
    """Stage 2: Verify _get_single_subject_data returns proper structure."""

    def test_raw_structure(self, name, module, cls_name):
        import mne

        dataset = _get_dataset(module, cls_name)
        subject = dataset.subject_list[0]
        data = dataset._get_single_subject_data(subject)

        # Must return {str: {str: mne.io.Raw}}
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        for session_key, session_data in data.items():
            assert isinstance(
                session_key, str
            ), f"Session key must be str, got {type(session_key)}"
            assert isinstance(session_data, dict), "Session data must be dict"
            for run_key, raw in session_data.items():
                assert isinstance(
                    run_key, str
                ), f"Run key must be str, got {type(run_key)}"
                assert isinstance(
                    raw, mne.io.BaseRaw
                ), f"Expected BaseRaw, got {type(raw)}"
                assert raw.n_times > 0, "Raw data must have samples"


@requires_download()
@pytest.mark.download
@pytest.mark.parametrize("name,module,cls_name", DATASETS, ids=[d[0] for d in DATASETS])
class TestStage3ParadigmGetData:
    """Stage 3: Verify SSVEP paradigm can extract epochs."""

    def test_paradigm_get_data(self, name, module, cls_name):
        from moabb.paradigms import SSVEP

        dataset = _get_dataset(module, cls_name)
        subject = dataset.subject_list[0]

        # Use broadband SSVEP paradigm
        paradigm = SSVEP(fmin=1, fmax=45, n_classes=None)

        X, labels, metadata = paradigm.get_data(dataset, subjects=[subject])

        assert X.shape[0] > 0, "Must have at least one epoch"
        assert len(labels) == X.shape[0], "Labels must match epochs"
        assert "subject" in metadata.columns, "Metadata must have subject column"
        assert "session" in metadata.columns, "Metadata must have session column"


@requires_download()
@pytest.mark.download
@pytest.mark.parametrize("name,module,cls_name", DATASETS, ids=[d[0] for d in DATASETS])
class TestStage4Evaluation:
    """Stage 4: End-to-end benchmark with simple pipeline."""

    def test_within_session_eval(self, name, module, cls_name):
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        from sklearn.pipeline import make_pipeline

        from moabb.evaluations import WithinSessionEvaluation
        from moabb.paradigms import SSVEP

        dataset = _get_dataset(module, cls_name)
        subject = dataset.subject_list[0]

        paradigm = SSVEP(fmin=1, fmax=45, n_classes=None)
        pipeline = make_pipeline(Covariances("oas"), MDM())

        evaluation = WithinSessionEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            overwrite=True,
        )

        results = evaluation.process(
            pipelines={"MDM": pipeline},
            subjects=[subject],
        )

        assert not results.empty, "Results DataFrame must not be empty"
        assert "score" in results.columns, "Results must have score column"
        scores = results["score"].values
        assert all(isinstance(s, float) for s in scores), "Scores must be floats"


# Standalone quick test for imports
class TestImports:
    """Verify all datasets can be imported."""

    def test_import_all(self):
        from ssvep_moabb_adapters import (
            Chen2017SingleFlicker,
            Dong2023,
            Han2024Fatigue,
            Kim2025BetaRange,
            Lee2021Mobile_SSVEP,
            Liu2020BETA,
            Liu2022EldBETA,
            Wang2021Combined,
        )

        # Verify they can be instantiated
        for cls in [
            Liu2020BETA,
            Liu2022EldBETA,
            Kim2025BetaRange,
            Dong2023,
            Han2024Fatigue,
            Chen2017SingleFlicker,
            Wang2021Combined,
            Lee2021Mobile_SSVEP,
        ]:
            dataset = cls()
            assert hasattr(dataset, "subject_list")
            assert hasattr(dataset, "event_id")
            assert len(dataset.subject_list) > 0
            assert len(dataset.event_id) > 0

    @pytest.mark.parametrize(
        "cls_name,expected_subjects,expected_events",
        [
            ("Liu2020BETA", 70, 40),
            ("Liu2022EldBETA", 100, 9),
            ("Kim2025BetaRange", 40, 40),
            ("Dong2023", 59, 40),
            ("Han2024Fatigue", 24, 32),
            ("Chen2017SingleFlicker", 12, 4),
            ("Wang2021Combined", 20, 4),
            ("Lee2021Mobile_SSVEP", 24, 3),
        ],
    )
    def test_dataset_properties(self, cls_name, expected_subjects, expected_events):
        import ssvep_moabb_adapters

        cls = getattr(ssvep_moabb_adapters, cls_name)
        dataset = cls()
        assert len(dataset.subject_list) == expected_subjects, (
            f"{cls_name}: expected {expected_subjects} subjects, "
            f"got {len(dataset.subject_list)}"
        )
        assert len(dataset.event_id) == expected_events, (
            f"{cls_name}: expected {expected_events} events, "
            f"got {len(dataset.event_id)}"
        )
