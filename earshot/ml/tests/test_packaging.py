import subprocess
import sys
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_BACKENDS = {"ai-edge-litert", "tflite-runtime"}


def selected_dependencies(machine, python_version):
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    environment = default_environment()
    environment.update(
        {
            "platform_machine": machine,
            "python_version": python_version,
            "python_full_version": python_version + ".0",
        }
    )
    selected = set()
    for declaration in data["project"]["dependencies"]:
        requirement = Requirement(declaration)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            selected.add(requirement.name)
    return selected


def test_pyproject_declares_package_and_cli():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["requires-python"] == ">=3.10,<3.15"
    assert data["project"]["scripts"]["earshot"] == "earshot_ml.cli:main"
    assert "test" in data["project"]["optional-dependencies"]
    assert (
        "tomli; python_version < '3.11'"
        in data["project"]["optional-dependencies"]["test"]
    )


def test_training_extra_is_optional_and_runtime_stays_lightweight():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert any(
        item.startswith("scikit-learn>=1.7")
        for item in data["project"]["optional-dependencies"]["train"]
    )
    assert all(
        not item.startswith("scikit-learn")
        for item in data["project"]["dependencies"]
    )


def test_alarm_training_import_does_not_require_scikit_learn():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['sklearn'] = None; "
                "import earshot_ml.alarm_training"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("machine", "python_version", "expected_backend"),
    [
        ("armv7l", "3.11", "tflite-runtime"),
        ("armv7l", "3.12", "tflite-runtime"),
        ("AMD64", "3.12", "ai-edge-litert"),
    ],
)
def test_runtime_backend_marker_matrix(machine, python_version, expected_backend):
    selected = selected_dependencies(machine, python_version)

    assert selected & RUNTIME_BACKENDS == {expected_backend}


def test_gitignore_excludes_python_build_artifacts():
    entries = {
        line.strip()
        for line in (ROOT.parent / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {"ml/build/", "ml/*.egg-info/"} <= entries


def test_gitignore_excludes_local_alarm_data_and_reports():
    entries = {
        line.strip()
        for line in (ROOT.parent / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {"ml/data/", "ml/models/*_report.json"} <= entries
