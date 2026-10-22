"""Keep the reusable APIs implemented inside their owning packages."""

import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("package_name", "export_name"),
    [
        ("core", "TranslationPipelinePaths"),
        ("core", "run_translation_pipeline"),
        ("core", "FilesystemSourceConnector"),
        ("formats", "LocalizationFormat"),
        ("formats", "LocalizationLayout"),
        ("formats", "LocalizationProfile"),
        ("formats", "LocalizationFileAdapter"),
        ("formats", "register_localization_adapter"),
        ("providers", "ChatModelProvider"),
        ("providers", "AiSuiteProvider"),
        ("providers", "OpenAICompatibleProvider"),
        ("providers", "create_model_provider"),
    ],
)
def test_public_api_implementation_lives_in_owning_package(package_name, export_name):
    package = importlib.import_module(f"localize.{package_name}")
    implementation = inspect.getmodule(getattr(package, export_name))

    assert implementation is not None
    assert implementation.__name__.startswith(f"localize.{package_name}.")
    assert Path(implementation.__file__).is_relative_to(Path(package.__file__).parent)


@pytest.mark.parametrize("package_name", ["core", "formats", "providers"])
def test_public_package_import_does_not_load_runtime_or_other_layers(package_name, tmp_path):
    # Run in a fresh interpreter so the global runtime fixtures cannot hide
    # circular imports, accidental config loading, or cross-layer dependencies.
    script = f"""
import importlib
import sys

package_name = {package_name!r}
importlib.import_module(f"localize.{{package_name}}")
for other in {{"core", "formats", "providers"}} - {{package_name}}:
    assert f"localize.{{other}}" not in sys.modules
assert "localize.app_config" not in sys.modules
assert "localize.translate_localization_files" not in sys.modules
assert "localize.guardian" not in sys.modules
"""
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["TRANSLATOR_CONFIG_FILE"] = str(tmp_path / "must-not-be-loaded.yaml")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
