"""Bounded private author dispositions; a decline is never a publication approval."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import stat

from localize.guardian.codex import CodexOutputError, _strict_json_loads

SCHEMA_PATH = Path(__file__).with_name("schemas") / "prevention-decision.schema.json"
DECLINE_REASONS = frozenset({"project_policy", "no_generic_defect", "already_covered"})
MAX_RESULT_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class PreventionDecision:
    outcome: str
    reason: str
    explanation: str
    inspected_paths: tuple[str, ...]


def _is_explanation_control(ch: str) -> bool:
    """Allow prose line breaks and tabs, but no terminal or other C0 controls."""
    return (ord(ch) < 32 and ch not in "\n\t") or ord(ch) == 127


def parse_decision(value: object) -> PreventionDecision:
    """Validate model output independently of the provider's schema enforcement."""
    try:
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "outcome",
            "reason",
            "explanation",
            "inspected_paths",
        }:
            raise ValueError
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError
        outcome, reason = value["outcome"], value["reason"]
        if not isinstance(outcome, str) or not isinstance(reason, str):
            raise ValueError
        if outcome == "patch":
            if reason != "":
                raise ValueError
        elif outcome != "declined" or reason not in DECLINE_REASONS:
            raise ValueError
        explanation, paths = value["explanation"], value["inspected_paths"]
        if (
            not isinstance(explanation, str)
            or not explanation.strip()
            or len(explanation) > 2000
            or any(_is_explanation_control(ch) for ch in explanation)
        ):
            raise ValueError
        if not isinstance(paths, list) or not 1 <= len(paths) <= 16:
            raise ValueError
        for path in paths:
            if (
                not isinstance(path, str)
                or not 1 <= len(path) <= 256
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in path)
                or "\\" in path
                or PurePosixPath(path).is_absolute()
                or any(part in {"", ".", "..", ".git"} for part in path.split("/"))
            ):
                raise ValueError
        if len(set(paths)) != len(paths):
            raise ValueError
        return PreventionDecision(outcome, reason, explanation, tuple(paths))
    except (TypeError, ValueError, UnicodeError):
        raise CodexOutputError("Prevention author disposition is invalid.") from None


def read_decision(path: Path, *, secrets: tuple[str, ...] = ()) -> PreventionDecision:
    """Read one required regular result file without following links or echoing data."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError
            raw = stream.read(MAX_RESULT_BYTES + 1)
        if len(raw) > MAX_RESULT_BYTES:
            raise ValueError
        text = raw.decode("utf-8")
        payload = _strict_json_loads(text)
        result = parse_decision(payload)
    except (OSError, ValueError, UnicodeError):
        raise CodexOutputError(
            "Prevention author disposition is missing or malformed."
        ) from None
    # Check decoded values as well as raw JSON so escaping cannot hide a secret.
    if any(
        secret
        and any(
            secret in v for v in (text, result.explanation, *result.inspected_paths)
        )
        for secret in secrets
    ):
        raise CodexOutputError("Prevention author disposition contains a credential.")
    return result
