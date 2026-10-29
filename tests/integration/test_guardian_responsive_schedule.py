"""Real schedule/state integration around the controller's I/O boundary."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest

from localize.guardian import runtime
from localize.guardian.controller import PollOutcome
from localize.guardian.credentials import CredentialError
from localize.guardian.models import GuardianMode
from localize.guardian.state import GuardianState
from tests.unit.test_guardian_controller import (
    FakeCodexDriver, FakeCurrentBaseProvider, FakeHistoricalCheckoutFactory,
    FakeHistoricalSnapshotProvider, _config, _controller, _feedback,
    _historical_policy, _snapshot, runtime as _controller_runtime_fixture,
)
from tests.unit.test_guardian_runtime import (
    NOW, _Credential, _stub_runtime_authority_for_assembly_tests as _authority_fixture,
    _write_minimal_config,
)

controller_runtime = _controller_runtime_fixture
_stub_runtime_authority_for_assembly_tests = _authority_fixture


@pytest.mark.parametrize("circuit", [None, "authentication_circuit_open", "model_circuit_open"])
def test_responsive_scheduler_retries_failure_but_stops_on_circuit(
    tmp_path, monkeypatch, circuit,
):
    config_path = tmp_path / "guardian.yaml"
    _write_minimal_config(config_path)
    config_path.write_text(
        "schedule: {poll_interval_seconds: 900, max_polls_per_day: 3}\n"
        + config_path.read_text(), encoding="utf-8",
    )
    clock = [NOW]
    calls = []

    class Controller:
        def poll_once(self, *, include_history=True):
            calls.append(clock[0])
            return PollOutcome(
                lease_acquired=True, runs_failed=1,
                **({circuit: True} if circuit else {}),
            )

    @contextmanager
    def credentials(*args, **kwargs):
        yield lambda: {}

    monkeypatch.setattr(runtime, "_local_now", lambda: clock[0])
    monkeypatch.setattr(runtime, "git_credential_environment", credentials)
    monkeypatch.setattr(runtime, "_build_controller", lambda **kwargs: Controller())
    assert runtime.run_once(config_path, scheduled=True) == 1
    assert runtime.run_once(config_path, scheduled=True) == 0
    for _ in range(4):
        clock[0] += timedelta(minutes=15)
        runtime.run_once(config_path, scheduled=True)
    assert len(calls) == (1 if circuit else 3)
    clock[0] += timedelta(days=1)
    assert runtime.run_once(config_path, scheduled=True) == 1
    assert len(calls) == (2 if circuit else 4)


def test_interval_poll_handles_new_feedback_without_replaying_history_or_model(
    tmp_path, controller_runtime,
):
    base, head, checkout, provider, broker, sequence = controller_runtime
    history = FakeHistoricalSnapshotProvider((), sequence=sequence)
    driver = FakeCodexDriver()
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path, state=state,
            config=_config(GuardianMode.OBSERVE, policies=(_historical_policy(),)),
            checkout=checkout, provider=provider, broker=broker, driver=driver,
            historical_snapshot_provider=history,
            historical_checkout_factory=FakeHistoricalCheckoutFactory(base, head, tmp_path),
            current_base_provider=FakeCurrentBaseProvider(),
        )
        first = controller.poll_once()
        assert first.historical_repositories_polled == 1
        assert len(driver.calls) == 1
        second = controller.poll_once(include_history=False)
        assert second.historical_repositories_polled == 0
        assert len(driver.calls) == 1
        provider.snapshots = (_snapshot(feedback=(_feedback(source_id="46"),)),)
        third = controller.poll_once(include_history=False)
        assert third.runs_failed == 0
        assert len(driver.calls) == 2
        assert third.historical_repositories_polled == 0
        assert controller.poll_once(include_history=False).runs_failed == 0
        assert len(driver.calls) == 2


def test_interval_schedule_limits_history_but_manual_poll_can_retry_it(tmp_path, monkeypatch):
    config_path = tmp_path / "guardian.yaml"
    _write_minimal_config(config_path)
    config_path.write_text(
        "schedule: {poll_interval_seconds: 900}\n" + config_path.read_text(), encoding="utf-8",
    )
    clock = [NOW]
    histories = []

    class Controller:
        def poll_once(self, *, include_history=True):
            histories.append(include_history)
            return PollOutcome(lease_acquired=True)

    @contextmanager
    def credentials(*args, **kwargs):
        yield lambda: {}

    monkeypatch.setattr(runtime, "_local_now", lambda: clock[0])
    monkeypatch.setattr(runtime, "git_credential_environment", credentials)
    monkeypatch.setattr(runtime, "_build_controller", lambda **kwargs: Controller())
    assert runtime.run_once(config_path, scheduled=True) == 0
    clock[0] += timedelta(minutes=15)
    assert runtime.run_once(config_path, scheduled=True) == 0
    assert runtime.run_once(config_path, scheduled=False) == 0
    assert histories == [True, False, True]


def test_responsive_scheduler_records_crashed_attempt_before_controller_call(
    tmp_path, monkeypatch,
):
    config_path = tmp_path / "guardian.yaml"
    _write_minimal_config(config_path)
    config_path.write_text(
        "schedule: {poll_interval_seconds: 900, max_polls_per_day: 1}\n"
        + config_path.read_text(), encoding="utf-8",
    )
    clock = [NOW]

    def failed_setup(*args, **kwargs):
        raise RuntimeError("bounded setup failure")

    monkeypatch.setattr(runtime, "_local_now", lambda: clock[0])
    monkeypatch.setattr(runtime, "_validate_runtime_authority", failed_setup)
    with pytest.raises(runtime.GuardianRuntimeError):
        runtime.run_once(config_path, scheduled=True)
    clock[0] += timedelta(hours=1)
    assert runtime.run_once(config_path, scheduled=True) == 0


def test_credential_failure_circuit_survives_failed_manual_recovery(
    tmp_path, monkeypatch,
):
    config_path = tmp_path / "guardian.yaml"
    _write_minimal_config(config_path)
    config_path.write_text(
        "schedule: {poll_interval_seconds: 900}\n" + config_path.read_text(), encoding="utf-8",
    )
    clock = [NOW]

    @contextmanager
    def failed_credentials(*args, **kwargs):
        raise CredentialError("operator broker is unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "_local_now", lambda: clock[0])
    monkeypatch.setattr(runtime, "_deadline_credential_snapshot", failed_credentials)
    with pytest.raises(runtime.GuardianRuntimeError):
        runtime.run_once(config_path, scheduled=True)
    clock[0] += timedelta(minutes=15)
    assert runtime.run_once(config_path, scheduled=True) == 0
    with pytest.raises(runtime.GuardianRuntimeError):
        runtime.run_once(config_path, scheduled=False)
    clock[0] += timedelta(minutes=15)
    assert runtime.run_once(config_path, scheduled=True) == 0


@pytest.mark.parametrize("circuit", ["authentication_circuit_open", "model_circuit_open"])
def test_only_successful_manual_recovery_clears_existing_circuit(tmp_path, monkeypatch, circuit):
    config_path = tmp_path / "guardian.yaml"
    _write_minimal_config(config_path)
    config_path.write_text(
        "schedule: {poll_interval_seconds: 900}\n" + config_path.read_text(), encoding="utf-8",
    )
    clock = [NOW]
    outcomes = iter([
        PollOutcome(lease_acquired=True, **{circuit: True}),
        PollOutcome(lease_acquired=True, runs_failed=1),
        PollOutcome(lease_acquired=True), PollOutcome(lease_acquired=True),
    ])

    class Controller:
        def poll_once(self, **kwargs):
            return next(outcomes)

    @contextmanager
    def credentials(*args, **kwargs):
        yield lambda: {}

    monkeypatch.setattr(runtime, "_local_now", lambda: clock[0])
    monkeypatch.setattr(runtime, "git_credential_environment", credentials)
    monkeypatch.setattr(runtime, "_build_controller", lambda **kwargs: Controller())
    assert runtime.run_once(config_path, scheduled=True) == 1
    clock[0] += timedelta(minutes=15)
    assert runtime.run_once(config_path, scheduled=False) == 1
    clock[0] += timedelta(minutes=15)
    assert runtime.run_once(config_path, scheduled=True) == 0
    assert runtime.run_once(config_path, scheduled=False) == 0
    clock[0] += timedelta(minutes=15)
    assert runtime.run_once(config_path, scheduled=True) == 0
    with pytest.raises(StopIteration):
        next(outcomes)


def test_missing_subscription_login_stops_repeated_scheduled_auth_attempts(tmp_path, monkeypatch):
    config_path = tmp_path / "guardian.yaml"
    _write_minimal_config(config_path)
    config_path.write_text(
        "schedule: {poll_interval_seconds: 900}\n" + config_path.read_text(), encoding="utf-8",
    )
    clock = [NOW]

    def require_subscription(config, **kwargs):
        runtime._validate_subscription_codex_home(replace(
            config, runtime=replace(config.runtime, codex_home=str(tmp_path / "missing-login")),
        ))

    monkeypatch.setattr(runtime, "_local_now", lambda: clock[0])
    monkeypatch.setattr(runtime, "_validate_runtime_authority", require_subscription)
    with pytest.raises(runtime.GuardianRuntimeError, match="ChatGPT authentication"):
        runtime.run_once(config_path, scheduled=True)
    clock[0] += timedelta(minutes=15)
    assert runtime.run_once(config_path, scheduled=True) == 0


@pytest.mark.parametrize("failure", ["connect", "timeout", "protocol", 502, 503, 429, "rate-limit"])
def test_preflight_network_failure_retries_same_day_after_restart(
    tmp_path, monkeypatch, controller_runtime, failure,
):
    """Real HTTP classification, controller and persisted hourly gate recover."""
    config_path = tmp_path / "guardian.yaml"
    _write_minimal_config(config_path)
    config_path.write_text(
        "schedule: {poll_interval_seconds: 3600, max_polls_per_day: 3}\n"
        + config_path.read_text(), encoding="utf-8",
    )
    _base, _head, checkout, provider, broker, _sequence = controller_runtime
    provider.snapshots = ()
    clock = [NOW]
    requests = []
    driver = FakeCodexDriver()
    policy = _config(GuardianMode.OBSERVE).repositories[0]

    def handler(request):
        assert request.method == "GET" and request.url.path == "/user"
        requests.append(clock[0])
        if len(requests) <= 2:
            errors = {"connect": httpx.ConnectError, "timeout": httpx.ReadTimeout,
                      "protocol": httpx.RemoteProtocolError}
            if failure in errors:
                raise errors[failure]("private transport detail", request=request)
            if failure == "rate-limit":
                return httpx.Response(403, headers={"Retry-After": "30"})
            return httpx.Response(failure)
        actor = policy.publication_actor
        return httpx.Response(200, json={"login": actor.login, "id": actor.id, "type": actor.type})

    actor_provider = runtime.AuthenticatedGitHubSnapshotProvider(
        credential=_Credential("test-only-token"), transport=httpx.MockTransport(handler),
    )

    def build(**kwargs):
        return _controller(
            tmp_path=tmp_path, state=kwargs["state"],
            config=replace(kwargs["config"], repositories=(policy,)),
            checkout=checkout, provider=provider, broker=broker, driver=driver,
            now=lambda: clock[0],
            publication_actor_preflight=lambda: actor_provider.require_publication_actor(
                policy, (policy.publication_actor,),
            ),
        )

    @contextmanager
    def credentials(*args, **kwargs):
        yield lambda: {}

    monkeypatch.setattr(runtime, "_local_now", lambda: clock[0])
    monkeypatch.setattr(runtime, "git_credential_environment", credentials)
    monkeypatch.setattr(runtime, "_build_controller", build)
    # Every invocation reopens the persisted state, as a new scheduler process does.
    for attempt in range(2):
        assert runtime.run_once(config_path, scheduled=True) == 1
        with GuardianState(tmp_path / ".guardian/state.sqlite3") as state:
            health = state.latest_health("guardian")
            checkpoint = state.latest_health("guardian-responsive-schedule")
            assert health.status == "failed"
            assert health.details["authentication_circuit_open"] is False
            assert health.details["repositories_polled"] == 0
            assert checkpoint.details["circuit_open"] is False
            assert checkpoint.details["attempt_count"] == attempt + 1
        assert provider.calls == []
        assert driver.calls == []
        assert broker.reply_calls == []
        clock[0] += timedelta(minutes=59)
        assert runtime.run_once(config_path, scheduled=True) == 0
        assert len(requests) == attempt + 1
        clock[0] += timedelta(minutes=1)
    assert runtime.run_once(config_path, scheduled=True) == 0
    assert len(requests) == 3
    with GuardianState(tmp_path / ".guardian/state.sqlite3") as state:
        assert state.latest_health("guardian").status == "ok"
        checkpoint = state.latest_health("guardian-responsive-schedule")
        assert checkpoint.details["circuit_open"] is False
        assert checkpoint.details["attempt_count"] == 3
    assert len(provider.calls) == 1
    clock[0] += timedelta(hours=1)
    assert runtime.run_once(config_path, scheduled=True) == 0
    assert len(requests) == 3  # The existing daily attempt budget is still enforced.
