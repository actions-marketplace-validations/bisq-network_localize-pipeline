"""Exercise stage selection at the CLI boundary with one durable call budget."""

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from localize.guardian import codex, prevention_runtime, runtime as production
from localize.guardian.models import CodexAuthMode, GuardianMode, RecurrenceCandidate
from localize.guardian.state import GuardianState
from tests.unit.test_guardian_controller import (
    NOW, FakeCodexDriver, _config, runtime,
)

controller_runtime = runtime


@pytest.mark.parametrize("auth_mode", list(CodexAuthMode))
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("prevention_failure", [False, True])
def test_assembled_stages_use_selected_models_and_share_call_budget(
    tmp_path, monkeypatch, controller_runtime, auth_mode, split, prevention_failure,
):
    """Real assembly, assessment, author driver and SQLite; fake only external I/O."""
    _base, _head, checkout, provider, broker, _sequence = controller_runtime
    config = _config(GuardianMode.PROPOSE_PREVENTION)
    overrides = (
        {"prevention_model": "gpt-6-astra", "prevention_reasoning_effort": "medium"}
        if split else {}
    )
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            codex_model="gpt-6-sol",
            codex_reasoning_effort="high",
            signing_key="A" * 40,
            codex_auth_mode=auth_mode,
            codex_api_key_command=(
                config.runtime.codex_api_key_command
                if auth_mode is CodexAuthMode.API_KEY else ()
            ),
            **overrides,
        ),
        limits=replace(
            config.limits, max_attempts=1, max_model_calls_per_day=2,
            daily_cost_limit_usd=2 if auth_mode is CodexAuthMode.API_KEY else None,
            model_call_reservation_usd=1 if auth_mode is CodexAuthMode.API_KEY else None,
        ),
    )
    invocations = []

    def process(argv, **kwargs):
        stage = "assessment" if 'default_permissions="guardian_evidence"' in argv else "prevention"
        model = argv[argv.index("--model") + 1]
        effort = next(x for x in argv if x.startswith("model_reasoning_effort="))
        invocations.append((stage, model, effort))
        for forbidden in ("GITHUB_TOKEN", "GH_TOKEN", "GIT_ASKPASS", "SSH_AUTH_SOCK"):
            assert forbidden not in kwargs["env"]
        assert ("CODEX_API_KEY" in kwargs["env"]) == (auth_mode is CodexAuthMode.API_KEY)
        if stage == "assessment":
            assert 'default_permissions="guardian_evidence"' in argv
            task = codex.CodexTask(
                prompt="fixture", evidence_dir=Path(argv[argv.index("-C") + 1]),
            )
            Path(argv[argv.index("-o") + 1]).write_text(
                codex.serialize_codex_result(FakeCodexDriver().run(task)),
            )
        else:
            assert 'default_permissions="guardian_prevention_author"' in argv
            if prevention_failure:
                return subprocess.CompletedProcess(argv, 1, "", "You've hit your usage limit")
        if stage == "prevention":
            Path(argv[argv.index("-o") + 1]).write_text(json.dumps({
                "schema_version": 1, "outcome": "patch", "reason": "",
                "explanation": "Implemented the generic engine correction.",
                "inspected_paths": ["localize/rules.py"]}))
        return subprocess.CompletedProcess(
            argv, 0,
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10,
                        "output_tokens": 2}, "cost_usd": 0.25}), "",
        )

    monkeypatch.setattr(codex, "run_bounded_process", process)
    monkeypatch.setattr(prevention_runtime, "run_bounded_process", process)
    monkeypatch.setenv("GH_TOKEN", "must-not-reach-model")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/must-not-reach-model")
    monkeypatch.setattr(production, "AuthenticatedGitHubSnapshotProvider", lambda **kw: provider)
    monkeypatch.setattr(production, "materialize_exact_checkout", lambda rev, **kw: checkout(rev))
    monkeypatch.setattr(production, "materialize_historical_checkout", lambda rev, **kw: checkout(rev))
    monkeypatch.setattr(production, "GitHubWriteBroker", lambda **kw: broker)
    monkeypatch.setattr(provider, "require_publication_actor", lambda *a: None, raising=False)
    monkeypatch.setattr(production, "resolve_model_api_key", lambda _: "test-only-model-key")

    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = production._build_controller(
            config=config, state=state, state_directory=tmp_path,
            github_credential=SimpleNamespace(), model_credential=None,
            git_environment=lambda: {},
        )
        controller.now = lambda: NOW
        coordinator = controller.prevention_runner
        coordinator.now = lambda: NOW
        outcome = controller.poll_once()
        assert outcome.runs_completed == 1
        assert outcome.runs_failed == 0
        assert state.model_calls_committed_for_day(NOW.date()) == 1
        run_id = state._connection.execute(
            "SELECT run_id FROM model_call_reservations",
        ).fetchone()[0]
        workspace = tmp_path / "author-workspace"
        workspace.mkdir()
        base_workspace = tmp_path / "author-base"
        base_workspace.mkdir()
        kwargs = dict(
            run_id=run_id, base_workspace=base_workspace, workspace=workspace,
            candidate=RecurrenceCandidate(
                scope="pipeline_code", summary="A recurring placeholder defect",
                evidence_feedback_ids=("review_comment:42",),
            ),
            evidence_ids=("review_comment:42",),
            policy=config.repositories[0].prevention,
            require_live_lease=lambda: None, require_cleanup_lease=lambda: None,
        )
        if prevention_failure:
            with pytest.raises(codex.CodexCapacityError):
                coordinator._reserve_and_author(**kwargs)
        else:
            coordinator._reserve_and_author(**kwargs)

        # The second stage cannot get a fresh budget simply by changing models.
        with pytest.raises(prevention_runtime.PreventionRuntimeError, match="Daily model call"):
            coordinator._reserve_and_author(**kwargs)
        prevention_model = "gpt-6-astra" if split else "gpt-6-sol"
        prevention_effort = "medium" if split else "high"
        assert invocations == [
            ("assessment", "gpt-6-sol", 'model_reasoning_effort="high"'),
            ("prevention", prevention_model, f'model_reasoning_effort="{prevention_effort}"'),
        ]
        calls = state._connection.execute(
            "SELECT purpose, model, status FROM model_call_reservations ORDER BY call_id",
        ).fetchall()
        assert [tuple(row) for row in calls] == [
            ("assessment", "gpt-6-sol", "completed"),
            ("prevention", prevention_model, "unknown" if prevention_failure else "completed"),
        ]
        if auth_mode is CodexAuthMode.API_KEY:
            assert state.cost_for_day(NOW.date()) == Decimal("0.25" if prevention_failure else "0.5")
            reservations = state._connection.execute(
                "SELECT model, status FROM budget_reservations ORDER BY reservation_id",
            ).fetchall()
            assert [tuple(row) for row in reservations] == [
                ("gpt-6-sol", "settled"),
                (prevention_model, "unknown" if prevention_failure else "settled"),
            ]
        else:
            assert state.cost_for_day(NOW.date()) == 0
            assert state._connection.execute(
                "SELECT COUNT(*) FROM budget_reservations",
            ).fetchone()[0] == 0
