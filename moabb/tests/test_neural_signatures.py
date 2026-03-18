"""Tests for moabb.analysis.neural_signatures."""

import tempfile
from pathlib import Path

import pytest

from moabb.datasets.fake import FakeDataset


# Skip entire module if plotly is not installed
plotly = pytest.importorskip("plotly")

from moabb.analysis.neural_signatures import (  # noqa: E402
    NeuralSignatureData,
    compute_cvep_signature,
    compute_erd_ers_signature,
    compute_erp_signature,
    compute_rstate_signature,
    compute_ssvep_signature,
    generate_neural_signature,
    get_plotly_colorscale,
    get_plotly_template,
    plot_cvep_interactive,
    plot_erd_ers_interactive,
    plot_erp_interactive,
    plot_rstate_interactive,
    plot_ssvep_interactive,
)


_PARADIGM_CONFIGS = {
    "p300": dict(
        dataset_kw=dict(
            paradigm="p300",
            event_list=("Target", "NonTarget"),
            channels=("C3", "Cz", "C4", "Fz", "Pz", "Oz"),
            sfreq=128,
            n_events=40,
            duration=60,
        ),
        paradigm_cls="P300",
        paradigm_kw={},
        compute_fn=compute_erp_signature,
        plot_fn=plot_erp_interactive,
        sig_paradigm="p300",
        sig_type="erp",
    ),
    "imagery": dict(
        dataset_kw=dict(
            paradigm="imagery",
            event_list=("left_hand", "right_hand"),
            channels=("C3", "Cz", "C4"),
            sfreq=128,
            n_events=30,
            duration=60,
        ),
        paradigm_cls="MotorImagery",
        paradigm_kw=dict(n_classes=2),
        compute_fn=compute_erd_ers_signature,
        plot_fn=plot_erd_ers_interactive,
        sig_paradigm="imagery",
        sig_type="erd_ers",
    ),
    "ssvep": dict(
        dataset_kw=dict(
            paradigm="ssvep",
            event_list=("13.0", "15.0"),
            channels=("O1", "Oz", "O2"),
            sfreq=256,
            n_events=30,
            duration=60,
        ),
        paradigm_cls="SSVEP",
        paradigm_kw=dict(n_classes=2),
        compute_fn=compute_ssvep_signature,
        plot_fn=plot_ssvep_interactive,
        sig_paradigm="ssvep",
        sig_type="psd_snr",
    ),
    "cvep": dict(
        dataset_kw=dict(
            paradigm="cvep",
            event_list=("1.0", "0.0"),
            channels=("O1", "Oz", "O2"),
            sfreq=256,
            n_events=30,
            duration=60,
        ),
        paradigm_cls="CVEP",
        paradigm_kw=dict(n_classes=2),
        compute_fn=compute_cvep_signature,
        plot_fn=plot_cvep_interactive,
        sig_paradigm="cvep",
        sig_type="cvep_response",
    ),
    "rstate": dict(
        dataset_kw=dict(
            paradigm="rstate",
            event_list=("eyes_open", "eyes_closed"),
            channels=("C3", "Cz", "C4", "Fz", "Pz", "Oz"),
            sfreq=128,
            n_events=20,
            duration=120,
        ),
        paradigm_cls="RestingStateToP300Adapter",
        paradigm_kw=dict(tmin=0, tmax=3, resample=128),
        sig_paradigm="rstate",
        sig_type="rstate_psd",
        compute_fn=compute_rstate_signature,
        plot_fn=plot_rstate_interactive,
    ),
}

_PARADIGM_IDS = list(_PARADIGM_CONFIGS.keys())


def _make_dataset(cfg):
    return FakeDataset(n_subjects=2, n_sessions=1, n_runs=1, **cfg["dataset_kw"])


def _get_paradigm_cls(name):
    """Import paradigm class by name string."""
    import moabb.paradigms as mp

    return getattr(mp, name)


def _load_epochs(cfg):
    ds = _make_dataset(cfg)
    cls = _get_paradigm_cls(cfg["paradigm_cls"])
    kw = dict(cfg["paradigm_kw"])
    # rstate needs events from dataset
    if cfg["paradigm_cls"] == "RestingStateToP300Adapter":
        kw["events"] = list(ds.event_id.keys())
    paradigm = cls(**kw)
    epochs, _, _ = paradigm.get_data(ds, subjects=[1], return_epochs=True)
    return epochs


class TestPlotlyStyle:
    def test_template(self):
        t = get_plotly_template()
        assert t is not None and t.layout is not None
        assert len(t.layout.colorway) > 0

    def test_colorscale(self):
        cs = get_plotly_colorscale()
        assert len(cs) == 5
        assert cs[0][0] == 0.0 and cs[-1][0] == 1.0


def test_neural_signature_data_creation():
    sig = NeuralSignatureData(
        paradigm="p300",
        dataset_name="T",
        dataset_code="t",
        signature_type="erp",
    )
    assert sig.paradigm == "p300"
    assert isinstance(sig.data, dict) and isinstance(sig.metadata, dict)


@pytest.mark.parametrize("paradigm", _PARADIGM_IDS)
def test_compute_signature(paradigm):
    cfg = _PARADIGM_CONFIGS[paradigm]
    epochs = _load_epochs(cfg)
    sig = cfg["compute_fn"](epochs)

    assert sig.paradigm == cfg["sig_paradigm"]
    assert sig.signature_type == cfg["sig_type"]
    assert list(sig.metadata["n_trials"].values())[0] > 0


def test_erp_sem_shape():
    cfg = _PARADIGM_CONFIGS["p300"]
    sig = cfg["compute_fn"](_load_epochs(cfg))
    for name in sig.data["event_names"]:
        if name in sig.data["evokeds"]:
            assert sig.data["sems"][name].shape == sig.data["evokeds"][name].shape


def test_erd_ers_channel_selection():
    cfg = _PARADIGM_CONFIGS["imagery"]
    epochs = _load_epochs(cfg)
    sig = cfg["compute_fn"](epochs)
    for ch in sig.metadata["ch_names"]:
        assert ch in epochs.ch_names


def test_ssvep_stimulus_frequencies():
    cfg = _PARADIGM_CONFIGS["ssvep"]
    sig = cfg["compute_fn"](_load_epochs(cfg))
    assert len(sig.data["stimulus_frequencies"]) > 0


def test_rstate_band_powers_sum():
    cfg = _PARADIGM_CONFIGS["rstate"]
    sig = cfg["compute_fn"](_load_epochs(cfg))
    for name, bp in sig.data["band_powers"].items():
        total = sum(bp.values())
        assert 90 < total < 110, f"Band powers for {name} sum to {total}"


@pytest.mark.parametrize("paradigm", _PARADIGM_IDS)
def test_plot_produces_figure(paradigm):
    cfg = _PARADIGM_CONFIGS[paradigm]
    sig = cfg["compute_fn"](_load_epochs(cfg))
    sig.dataset_name = "Test"

    fig = cfg["plot_fn"](sig)
    assert fig is not None
    assert len(fig.data) > 0


def test_erp_html_output():
    cfg = _PARADIGM_CONFIGS["p300"]
    sig = cfg["compute_fn"](_load_epochs(cfg))
    sig.dataset_name = "Test"
    html = cfg["plot_fn"](sig).to_html(include_plotlyjs=True)
    assert "plotly" in html.lower() and len(html) > 100


@pytest.mark.parametrize("paradigm", _PARADIGM_IDS)
def test_generate_end_to_end(paradigm):
    cfg = _PARADIGM_CONFIGS[paradigm]
    ds = _make_dataset(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_neural_signature(ds, subjects=[1], output_dir=tmpdir)
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()
            assert "plotly" in p.read_text().lower()


def test_output_dir_created():
    ds = _make_dataset(_PARADIGM_CONFIGS["p300"])
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "nested" / "dir"
        generate_neural_signature(ds, subjects=[1], output_dir=out)
        assert out.exists()


def test_unsupported_paradigm_raises():
    ds = FakeDataset(paradigm="imagery")
    ds._paradigm = "unknown"
    object.__setattr__(ds, "paradigm", "unknown")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unsupported paradigm"):
            generate_neural_signature(ds, subjects=[1], output_dir=tmpdir)


def test_all_paradigms_have_handlers():
    from moabb.analysis.neural_signatures import _PARADIGM_HANDLERS

    assert set(_PARADIGM_HANDLERS.keys()) == {
        "imagery",
        "p300",
        "ssvep",
        "cvep",
        "rstate",
    }


def test_handler_tuple_structure():
    from moabb.analysis.neural_signatures import _PARADIGM_HANDLERS

    for name, handler in _PARADIGM_HANDLERS.items():
        assert len(handler) == 2, f"{name}: expected 2-tuple"
        assert callable(handler[0]) and callable(handler[1])


@pytest.mark.parametrize("paradigm", _PARADIGM_IDS)
def test_head_svg_present_all_paradigms(paradigm):
    """Regression: head SVG diagram must appear for every paradigm."""
    cfg = _PARADIGM_CONFIGS[paradigm]
    ds = _make_dataset(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_neural_signature(ds, subjects=[1], output_dir=tmpdir)
        assert len(paths) >= 1
        html = paths[0].read_text()
        assert 'id="head-svg"' in html, f"head SVG missing for {paradigm}"
