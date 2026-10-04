from importlib.resources import as_file, files
from pathlib import Path

import pytest

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
