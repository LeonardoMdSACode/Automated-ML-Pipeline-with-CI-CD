'''New model version increments correctly'''
# tests\unit\test_version_increment.py

from pathlib import Path
import re
import pytest

def get_next_model_version(registry_dir: Path) -> str:
    """
    Returns next model version string (e.g. model_v003)
    based on existing directories in registry.
    """
    versions = []

    for p in registry_dir.iterdir():
        if p.is_dir():
            m = re.match(r"model_v(\d+)", p.name)
            if m:
                versions.append(int(m.group(1)))

    next_version = max(versions, default=0) + 1
    return f"model_v{next_version:03d}"

@pytest.mark.unit
def test_version_increment_empty_registry(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()

    version = get_next_model_version(registry)

    assert version == "model_v001"


@pytest.mark.unit
def test_version_increment_single_model(tmp_path):
    registry = tmp_path / "registry"
    (registry / "model_v001").mkdir(parents=True)

    version = get_next_model_version(registry)

    assert version == "model_v002"


@pytest.mark.unit
def test_version_increment_multiple_models(tmp_path):
    registry = tmp_path / "registry"
    for v in ["model_v001", "model_v002", "model_v007"]:
        (registry / v).mkdir(parents=True)

    version = get_next_model_version(registry)

    assert version == "model_v008"


@pytest.mark.unit
def test_version_increment_ignores_invalid_dirs(tmp_path):
    registry = tmp_path / "registry"
    for v in ["model_v001", "foo", "model_latest", "model_vabc"]:
        (registry / v).mkdir(parents=True)

    version = get_next_model_version(registry)

    assert version == "model_v002"
