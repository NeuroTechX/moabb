import pickle
import textwrap
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import pytest
from joblib import Parallel, delayed
from sklearn.base import clone

from moabb.datasets.fake import FakeDataset
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms.motor_imagery import FakeImageryParadigm
from moabb.pipelines import get_benchmark_pipelines
from moabb.pipelines.utils import parse_pipelines_from_directory


def test_reference_pipeline_resources_are_shipped():
    config_dir = files("moabb.pipelines").joinpath("configs")
    resources = [
        resource
        for resource in config_dir.iterdir()
        if resource.name != "__init__.py" and resource.name.endswith((".yml", ".py"))
    ]
    assert len(resources) == 28


def test_get_benchmark_pipelines_loads_packaged_configs():
    configs = get_benchmark_pipelines()
    assert len(configs) == 28
    assert all({"name", "pipeline", "paradigms"} <= config.keys() for config in configs)


def test_get_benchmark_pipelines_filters_by_paradigm():
    configs = get_benchmark_pipelines(paradigm="SSVEP")
    assert configs
    assert all("SSVEP" in config["paradigms"] for config in configs)


def test_parse_pipelines_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No pipeline configuration files"):
        parse_pipelines_from_directory(tmp_path)


def test_parse_single_python_pipeline_config():
    config = files("moabb.pipelines").joinpath("configs", "FBCSP.py")

    with as_file(config) as config_path:
        configs = parse_pipelines_from_directory(config_path)
    assert len(configs) == 1
    assert configs[0]["name"] == "FBCSP + optSVM"


def test_packaged_reference_configs_match_repository_sources():
    """Keep packaged mirrors byte-identical to the canonical repo configs."""

    package_dir = files("moabb.pipelines").joinpath("configs")
    repository_dir = Path(__file__).resolve().parents[2] / "pipelines"

    packaged = sorted(
        resource.name
        for resource in package_dir.iterdir()
        if resource.name != "__init__.py" and resource.name.endswith((".yml", ".py"))
    )
    canonical = sorted(
        path.name
        for path in repository_dir.iterdir()
        if path.name.endswith((".yml", ".py"))
    )

    assert packaged == canonical
    for name in canonical:
        with as_file(package_dir.joinpath(name)) as packaged_path:
            assert packaged_path.read_bytes() == (repository_dir / name).read_bytes(), (
                name
            )


# A Python pipeline config that defines its own function and class, as users
# do for small transforms (e.g. ``FunctionTransformer(_zscore)``).
_CONFIG_WITH_OWN_OBJECTS = textwrap.dedent(
    """
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer


    def _log_var(X):
        return np.log(np.var(X, axis=-1) + OFFSET)


    class Scale(BaseEstimator, TransformerMixin):
        def __init__(self, factor=2.0):
            self.factor = factor

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X * self.factor


    PIPELINE = {
        "name": "NAME",
        "paradigms": ["FakeImageryParadigm"],
        "pipeline": make_pipeline(
            FunctionTransformer(_log_var), Scale(), LinearDiscriminantAnalysis()
        ),
    }
    """
)


def _write_config(path, name, offset=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _CONFIG_WITH_OWN_OBJECTS.replace("OFFSET", repr(offset)).replace("NAME", name)
    )
    return path


def _toy_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3, 50))
    X[10:, 0] *= 3
    return X, np.repeat([0, 1], 10)


def _fit_predict(pipeline, X, y):
    return pipeline.fit(X, y).predict(X)


def test_python_pipeline_config_objects_survive_pickle(tmp_path):
    # Regression: .py configs ran as an unregistered module named "custom",
    # so pickling their functions/classes failed (PicklingError: import of
    # module 'custom' failed), e.g. in save_model.
    (config,) = parse_pipelines_from_directory(
        _write_config(tmp_path / "logvar_pipeline.py", "logvar")
    )
    pipeline = config["pipeline"]
    X, y = _toy_data()

    restored = pickle.loads(pickle.dumps(pipeline))
    assert restored[0].func is pipeline[0].func
    assert type(restored[1]) is type(pipeline[1])
    np.testing.assert_array_equal(
        _fit_predict(restored, X, y), _fit_predict(pipeline, X, y)
    )
    fitted = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(fitted.predict(X), pipeline.predict(X))


def test_python_pipeline_configs_sharing_a_file_name_stay_distinct(tmp_path):
    first = _write_config(tmp_path / "a" / "pipeline.py", "first", offset=1.0)
    second = _write_config(tmp_path / "b" / "pipeline.py", "second", offset=100.0)
    configs = parse_pipelines_from_directory(first) + parse_pipelines_from_directory(
        second
    )
    X, _ = _toy_data()

    assert configs[0]["pipeline"][0].func is not configs[1]["pipeline"][0].func
    for config, offset in zip(configs, (1.0, 100.0)):
        func = pickle.loads(pickle.dumps(config["pipeline"]))[0].func
        np.testing.assert_allclose(func(X), np.log(np.var(X, axis=-1) + offset))


def test_python_pipeline_config_parsed_twice_keeps_both_picklable(tmp_path):
    path = _write_config(tmp_path / "pipeline.py", "twice")
    (first,) = parse_pipelines_from_directory(path)
    (second,) = parse_pipelines_from_directory(path)

    assert first["pipeline"][0].func is not second["pipeline"][0].func
    for config in (first, second):
        restored = pickle.loads(pickle.dumps(config["pipeline"]))
        assert restored[0].func is config["pipeline"][0].func


def test_python_pipeline_config_reaches_joblib_workers(tmp_path):
    # Worker processes cannot import the config module: its objects must still
    # reach them by value (n_jobs > 1 in the evaluations).
    (config,) = parse_pipelines_from_directory(
        _write_config(tmp_path / "pipeline.py", "workers")
    )
    X, y = _toy_data()
    expected = _fit_predict(clone(config["pipeline"]), X, y)

    predictions = Parallel(n_jobs=2, backend="loky")(
        delayed(_fit_predict)(config["pipeline"], X, y) for _ in range(2)
    )
    for predicted in predictions:
        np.testing.assert_array_equal(predicted, expected)


def test_python_pipeline_config_model_is_saved(tmp_path):
    (config,) = parse_pipelines_from_directory(
        _write_config(tmp_path / "configs" / "pipeline.py", "saved")
    )
    dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=1, n_sessions=1, seed=3)
    evaluation = WithinSessionEvaluation(
        paradigm=FakeImageryParadigm(),
        datasets=[dataset],
        hdf5_path=str(tmp_path / "results"),
        save_model=True,
        overwrite=True,
    )

    results = evaluation.process({config["name"]: config["pipeline"]})

    assert len(results) == 1
    saved = sorted((tmp_path / "results").rglob("fitted_model_*.pkl"))
    assert saved
    with open(saved[0], "rb") as file:
        model = pickle.load(file)
    assert model[0].func is config["pipeline"][0].func
