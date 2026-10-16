"""Exercise the production translation fetch commands against real Git remotes."""

import re
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(command, *, cwd, check=True):
    return subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=check
    )


def _fetch_commands():
    commands = []
    for path in ("docker/docker-entrypoint.sh", "update-translations.sh"):
        script = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        commands.extend(
            line.removesuffix(" \\")
            for line in re.findall(r"^\s*(git fetch[^\n]+)", script, re.MULTILINE)
        )
    return commands


@pytest.fixture
def retagged_remote(tmp_path):
    remote = tmp_path / "remote.git"
    publisher = tmp_path / "publisher"
    target = tmp_path / "target"
    _run(["git", "init", "--bare", str(remote)], cwd=tmp_path)
    _run(["git", "init", "-b", "main", str(publisher)], cwd=tmp_path)
    _run(["git", "config", "user.name", "Fetch Test"], cwd=publisher)
    _run(["git", "config", "user.email", "fetch@example.invalid"], cwd=publisher)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=publisher)
    _run(["git", "commit", "--allow-empty", "-m", "Original release"], cwd=publisher)
    original = _run(["git", "rev-parse", "HEAD"], cwd=publisher).stdout.strip()
    _run(["git", "tag", "v2.1.14"], cwd=publisher)
    _run(["git", "remote", "add", "origin", str(remote)], cwd=publisher)
    _run(["git", "push", "origin", "main", "--tags"], cwd=publisher)
    _run(["git", "clone", "-b", "main", str(remote), str(target)], cwd=tmp_path)
    # A persisted repository may already have tags enabled; CLI must override it.
    _run(["git", "config", "remote.origin.tagOpt", "--tags"], cwd=target)
    _run(["git", "commit", "--allow-empty", "-m", "Updated branch"], cwd=publisher)
    latest = _run(["git", "rev-parse", "HEAD"], cwd=publisher).stdout.strip()
    _run(["git", "tag", "-f", "v2.1.14"], cwd=publisher)
    _run(["git", "push", "--force", "origin", "main", "--tags"], cwd=publisher)
    return target, original, latest


@pytest.mark.parametrize("fetch_command", _fetch_commands())
def test_translation_fetch_refreshes_branch_despite_retagged_release(
    retagged_remote, fetch_command
):
    target, original, latest = retagged_remote
    result = _run(
        [
            "bash", "-c",
            'set -euo pipefail; REMOTE=origin; DEFAULT_BRANCH=main; '
            + fetch_command,
        ],
        cwd=target,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _run(["git", "rev-parse", "origin/main"], cwd=target).stdout.strip() == latest
    assert _run(["git", "rev-parse", "v2.1.14"], cwd=target).stdout.strip() == original


@pytest.mark.parametrize("fetch_command", _fetch_commands()[:2])
def test_required_translation_fetch_stops_on_unreachable_remote(tmp_path, fetch_command):
    _run(["git", "init"], cwd=tmp_path)
    _run(["git", "remote", "add", "origin", str(tmp_path / "missing.git")], cwd=tmp_path)
    result = _run(
        [
            "bash", "-c",
            'set -euo pipefail; REMOTE=origin; DEFAULT_BRANCH=main; '
            + fetch_command + '; printf "unexpected continuation"',
        ],
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode != 0
    assert "unexpected continuation" not in result.stdout
