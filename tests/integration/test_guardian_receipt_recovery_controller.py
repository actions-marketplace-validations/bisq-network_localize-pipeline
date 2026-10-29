"""Receipt acknowledgement cannot process new feedback or bypass local authority."""

import pytest

from localize.guardian.models import GuardianMode
from localize.guardian.state import GuardianState
from tests.integration.test_guardian_feedback_reporting import DecisionDriver
from tests.unit.test_guardian_controller import _config, _controller, _feedback, _snapshot
from tests.unit.test_guardian_controller import runtime as _runtime_fixture

runtime = _runtime_fixture


def test_existing_receipt_leaves_new_feedback_for_intake(tmp_path, runtime, monkeypatch):
    _, _, checkout, provider, broker, _ = runtime
    driver = DecisionDriver()
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(tmp_path=tmp_path, state=state,
                                 config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
                                 checkout=checkout, provider=provider, driver=driver, broker=broker)
        assert controller.poll_once().failures == ()
        original_summary = broker.feedback_summary
        original_source_check = controller._require_exact_open_source_authority

        def forbidden_checkout(**kwargs):
            raise AssertionError("An exact existing receipt needs no source checkout")

        def recover_summary(**kwargs):
            kwargs["before_recover"]()
            provider.snapshots = (_snapshot(feedback=(_feedback(), _feedback(source_id="45", body="A new actionable suggestion."))),)
            kwargs["before_recover"]()
            return original_summary

        monkeypatch.setattr(controller, "_require_exact_open_source_authority", forbidden_checkout)
        broker.post_feedback_summary = recover_summary
        second = controller.poll_once()
        assert second.failures == () and second.runs_started == 0
        assert len(driver.calls) == 1
        assert len(state.latest_event_revisions()) == 1

        # The next intake must assess the new object. A changed summary is a
        # write, so it must pass the full current-source check again.
        full_checks = []
        def require_full(**kwargs):
            full_checks.append(True)
            return original_source_check(**kwargs)
        def update_summary(**kwargs):
            kwargs["before_create"]()
            return original_summary
        monkeypatch.setattr(controller, "_require_exact_open_source_authority", require_full)
        broker.post_feedback_summary = update_summary
        third = controller.poll_once()
        assert third.failures == () and len(driver.calls) == 2
        assert len(state.latest_event_revisions()) == 2
        new_revision = next(item for item in state.latest_event_revisions() if item.event_id == "45")
        assert state.latest_feedback_report(new_revision.revision_id)["verdict"] == "needs_human"
        assert full_checks


@pytest.mark.parametrize("revoke", ["intent", "lease_before", "lease_after"])
def test_receipt_acknowledgement_requires_local_intent_and_live_lease(tmp_path, runtime, monkeypatch, revoke):
    _, _, checkout, provider, broker, _ = runtime
    driver = DecisionDriver()
    broker.report_error = RuntimeError("response lost after creation")
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(tmp_path=tmp_path, state=state,
                                 config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
                                 checkout=checkout, provider=provider, driver=driver, broker=broker)
        assert controller.poll_once().failures == ("RuntimeError",)
        assert state.feedback_reporting_counts()[0] == 1
        broker.report_error = None

        def receipt(**kwargs):
            kwargs["before_recover"]()
            if revoke == "lease_after":
                monkeypatch.setattr(state, "refresh_lease", lambda **unused: False)
            return broker.feedback_reports[kwargs["report_id"]]

        broker.post_feedback_report = receipt
        if revoke == "intent":
            def invalid_attestation(**kwargs):
                raise ValueError("Local exact-source attestation changed")
            monkeypatch.setattr(state, "validate_prevention_source_attestation", invalid_attestation)
        elif revoke == "lease_before":
            monkeypatch.setattr(state, "refresh_lease", lambda **unused: False)
        result = controller.poll_once()
        assert result.failures
        assert state.feedback_reporting_counts()[0] == 1
        assert len(broker.feedback_reports) == 1 and len(driver.calls) == 1
