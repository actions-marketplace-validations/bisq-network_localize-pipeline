"""Explicit author dispositions must not turn unexplained empty work green."""

import json

import pytest

from localize.guardian.codex import CodexOutputError
from localize.guardian.prevention_decision import read_decision


def payload(**overrides):
    return dict(
        schema_version=1,
        outcome="declined",
        reason="project_policy",
        explanation="This requires a project-specific wording policy.",
        inspected_paths=["localize/rules.py"],
        **overrides,
    )


def test_read_explicit_decline(tmp_path):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(payload()))
    result = read_decision(p)
    assert result.outcome == "declined"
    assert result.reason == "project_policy"
    assert result.inspected_paths == ("localize/rules.py",)


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": ""},
        {"reason": "unexplained"},
        {"explanation": "  "},
        {"explanation": "x" * 2001},
        {"inspected_paths": []},
        {"inspected_paths": ["../outside"]},
        {"inspected_paths": ["/etc/passwd"]},
        {"inspected_paths": ["localize/rules.py", "localize/rules.py"]},
        {"schema_version": True},
        {"outcome": "success"},
        {"extra": 1},
        {"outcome": "patch", "reason": "project_policy"},
        {"explanation": "hidden\u001b[2J"},
        {"explanation": "carriage\rreturn"},
        {"explanation": "delete\u007f"},
        {"explanation": "\n\t "},
        {"inspected_paths": [""]},
        {"inspected_paths": ["x" * 257]},
        {"inspected_paths": [f"localize/f{i}.py" for i in range(17)]},
        {"inspected_paths": ["localize/rules.py\n"]},
        {"inspected_paths": ["localize/\trules.py"]},
    ],
)
def test_reject_malformed_decision(tmp_path, changes):
    p = tmp_path / "result.json"
    data = payload()
    data.update(changes)
    p.write_text(json.dumps(data))
    with pytest.raises(CodexOutputError):
        read_decision(p)


@pytest.mark.parametrize(
    "explanation",
    ["First line.\nSecond line.", "Column\tvalue", "x" * 2000],
)
def test_accept_multiline_and_tabbed_explanation(tmp_path, explanation):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(payload() | {"explanation": explanation}))
    assert read_decision(p).explanation == explanation


@pytest.mark.parametrize(
    "raw", ["", "not json", '{"outcome":"patch","outcome":"declined"}', '"x"']
)
def test_reject_missing_or_ambiguous_result(tmp_path, raw):
    p = tmp_path / "result.json"
    p.write_text(raw)
    with pytest.raises(CodexOutputError):
        read_decision(p)


def test_reject_symlink_and_oversized_results(tmp_path):
    p = tmp_path / "result.json"
    with pytest.raises(CodexOutputError):
        read_decision(p)
    other = tmp_path / "other"
    other.write_text(json.dumps(payload()))
    p.symlink_to(other)
    with pytest.raises(CodexOutputError):
        read_decision(p)
    p.unlink()
    p.write_text(" " * 16385)
    with pytest.raises(CodexOutputError):
        read_decision(p)


def test_result_cannot_persist_model_credential(tmp_path):
    p = tmp_path / "result.json"
    data = payload()
    data["explanation"] = "secret-token"
    p.write_text(json.dumps(data))
    with pytest.raises(CodexOutputError, match="credential"):
        read_decision(p, secrets=("secret-token",))
